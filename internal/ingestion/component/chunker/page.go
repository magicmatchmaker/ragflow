//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

// PageChunker emits one chunk per upstream slide or page.
// It is the faithful Go port of the Python `presentation` chunk method
// (rag/app/presentation.py), whose docstring states: "Every page will
// be treated as a chunk."
//
// Unlike TokenChunker (which merges slides into a single chunk) or
// OneChunker (which collapses many slides into one), PageChunker
// preserves the upstream slide records, including structured tables and
// images. Every record retains its slide_number for source-page identity.
package chunker

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"go.uber.org/zap"

	"ragflow/internal/agent/runtime"
	"ragflow/internal/common"
	deepdoctype "ragflow/internal/deepdoc/parser/type"
	"ragflow/internal/ingestion/component/schema"

	"gorm.io/gorm"
)

type pageChunkerParam struct{}

func (p *pageChunkerParam) Update(conf map[string]any) {}

func (pageChunkerParam) Defaults() pageChunkerParam {
	return pageChunkerParam{}
}

func (pageChunkerParam) Validate() error { return nil }

type PageChunkerComponent struct {
	name  string
	param pageChunkerParam
}

func NewPageChunker(params map[string]any) (runtime.Component, error) {
	p := pageChunkerParam{}.Defaults()
	(&p).Update(params)
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return &PageChunkerComponent{
		name:  ComponentNamePageChunker,
		param: p,
	}, nil
}
func (c *PageChunkerComponent) Inputs() map[string]string { return ChunkerInputs }

func (c *PageChunkerComponent) Outputs() map[string]string { return ChunkerOutputs }

func (c *PageChunkerComponent) Invoke(ctx context.Context, db *gorm.DB, inputs map[string]any) (map[string]any, error) {
	return c.invoke(ctx, db, inputs)
}

// invoke emits one chunk per decoded upstream slide or page.
func (c *PageChunkerComponent) invoke(ctx context.Context, db *gorm.DB, inputs map[string]any) (map[string]any, error) {
	if inputs == nil {
		return emptyOutputs(), nil
	}
	upstream, err := decodeChunkerFromUpstream(inputs)
	if err != nil {
		return inputErrorOutput(err), nil
	}

	items := slideItems(upstream.JSONResult, upstream.Chunks)
	if len(items) == 0 {
		return emptyOutputs(), nil
	}
	isPDF := isPDFPresentation(upstream)
	chunks := mergePresentationPages(ctx, items, !isPDF)
	if isPDF {
		chunks = attachPDFPagePreviews(ctx, db, upstream, chunks)
	}
	return chunkOutputs(chunks), nil
}

// slideItems returns the per-slide records, preferring JSONResult and
// falling back to Chunks. Each record (slide/page) becomes one chunk.
func slideItems(items, chunks []schema.ChunkDoc) []schema.ChunkDoc {
	if len(items) > 0 {
		return items
	}
	return chunks
}

// mergePresentationPages consolidates the parser's text, table and image
// records into the Presentation contract: one retrievable chunk per slide or
// page. Items without page identity retain their historical one-item
// granularity instead of being accidentally merged into one document chunk.
func mergePresentationPages(ctx context.Context, items []schema.ChunkDoc, enableOCR bool) []schema.ChunkDoc {
	type pageGroup struct {
		page  int
		items []schema.ChunkDoc
	}
	groups := make([]pageGroup, 0, len(items))
	byPage := make(map[int]int)
	for i := range items {
		page, identified := presentationPageNumber(items[i])
		if !identified {
			groups = append(groups, pageGroup{items: []schema.ChunkDoc{items[i]}})
			continue
		}
		groupIndex, ok := byPage[page]
		if !ok {
			groupIndex = len(groups)
			byPage[page] = groupIndex
			groups = append(groups, pageGroup{page: page})
		}
		groups[groupIndex].items = append(groups[groupIndex].items, items[i])
	}

	out := make([]schema.ChunkDoc, 0, len(groups))
	var ocrAnalyzer deepdoctype.DocAnalyzer
	var ocrAnalyzerErr error
	ocrAnalyzerResolved := false
	for _, group := range groups {
		merged := group.items[0]
		var textParts []string
		var image, imgID string
		var mediaRef = merged.MediaRef
		for _, item := range group.items {
			if text := strings.TrimSpace(item.Text); text != "" {
				textParts = append(textParts, text)
			}
			if image == "" {
				image = item.Image
			}
			if imgID == "" {
				imgID = item.ImgID
			}
			if mediaRef == nil {
				mediaRef = item.MediaRef
			}
		}
		merged.Text = strings.Join(textParts, "\n\n")
		merged.Image, merged.ImgID, merged.MediaRef = image, imgID, mediaRef
		merged.TableData = nil

		previewItems := group.items
		fallbackText := ""
		if len(textParts) == 0 && group.page > 0 {
			fallbackText = fmt.Sprintf("[Page %d]", group.page)
			// The synthetic label is preview text only. It must not inherit the
			// first source item's media or geometry: doing so would redraw that
			// image on top of later z-order items and could cover the full slide
			// with the text backing rectangle.
			fallback := schema.ChunkDoc{Text: fallbackText, DocType: "text"}
			previewItems = append(append([]schema.ChunkDoc(nil), group.items...), fallback)
		}
		preview, mediaCount, previewErr := composePresentationPreview(ctx, previewItems)
		if previewErr != nil {
			common.Warn("PageChunker: presentation preview was partially or completely unavailable",
				zap.Int("page", group.page), zap.Int("media_items", mediaCount), zap.Error(previewErr))
		}
		if len(preview) > 0 {
			stored, err := storePresentationPreview(ctx, preview)
			if err != nil {
				common.Warn("PageChunker: presentation preview cache write failed; retaining searchable text",
					zap.Int("page", group.page), zap.Int("media_items", mediaCount), zap.Error(err))
				merged.Image, merged.ImgID, merged.MediaRef = "", "", nil
			} else if stored != nil {
				merged.Image, merged.ImgID, merged.MediaRef = stored.Image, "", stored.MediaRef
			}
		} else if mediaCount > 0 {
			// Invalid or unsupported slide media must not turn a recoverable deck
			// into a failed ingestion. The merged searchable text remains.
			merged.Image, merged.ImgID, merged.MediaRef = "", "", nil
		}
		if enableOCR && mediaCount > 0 && len(preview) > 0 {
			if !ocrAnalyzerResolved {
				ocrAnalyzer, ocrAnalyzerErr = resolvePresentationOCRAnalyzer()
				ocrAnalyzerResolved = true
				if ocrAnalyzerErr != nil {
					common.Warn("PageChunker: local presentation OCR is unavailable; retaining parser text and preview", zap.Error(ocrAnalyzerErr))
				}
			}
			if ocrAnalyzer != nil {
				ocrText, err := extractPresentationOCR(ctx, preview, ocrAnalyzer)
				if err != nil {
					common.Warn("PageChunker: local presentation OCR failed; retaining parser text and preview",
						zap.Int("page", group.page), zap.Error(err))
				} else {
					merged.Text = mergePresentationOCR(merged.Text, ocrText)
				}
			}
		}
		if merged.Text == "" {
			merged.Text = fallbackText
		}
		if merged.Image != "" || merged.ImgID != "" || merged.MediaRef != nil {
			merged.DocType = "image"
		} else {
			merged.DocType = "text"
		}
		if group.page > 0 {
			page := group.page
			merged.PageNumber = &page
			_ = merged.SetExtraValue("slide_number", page)
		}
		clearPresentationPreviewMetadata(&merged)
		out = append(out, merged)
	}
	return out
}

func presentationPageNumber(item schema.ChunkDoc) (int, bool) {
	if raw, ok := item.Extra["slide_number"]; ok {
		var value any
		if json.Unmarshal(raw, &value) == nil {
			switch number := value.(type) {
			case float64:
				if number > 0 {
					return int(number), true
				}
			case string:
				var page int
				if _, err := fmt.Sscanf(number, "%d", &page); err == nil && page > 0 {
					return page, true
				}
			}
		}
	}
	if item.PageNumber != nil && *item.PageNumber > 0 {
		return *item.PageNumber, true
	}
	return 0, false
}

func isPDFPresentation(upstream schema.ChunkerFromUpstream) bool {
	fileType := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(upstream.FileType)), ".")
	return fileType == "pdf" || strings.HasSuffix(strings.ToLower(upstream.Name), ".pdf")
}

func init() {
	MustRegisterChunker(ComponentNamePageChunker)
}
