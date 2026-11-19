// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package chunker

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"ragflow/internal/agent/runtime"
	"ragflow/internal/common"
	"ragflow/internal/ingestion/component/schema"

	"gorm.io/gorm"
)

const (
	doclingPresentationPreviewSourceKey = "docling_presentation_preview_source"
	doclingSlidePreviewKey              = "docling_slide_preview"
	doclingImageRoleKey                 = "docling_image_role"
	doclingImageRoleFullSlide           = "full_slide"
	doclingPreviewAutoDisplayKey        = "docling_preview_auto_display"
	doclingPreviewReasonKey             = "docling_preview_display_reason"
	doclingPreviewReasonCover           = "cover_or_low_information"
	doclingPreviewReasonTOC             = "table_of_contents"
	doclingPreviewReasonContent         = "content_slide"
)

var doclingTOCMarkers = []string{
	"목차", "차례", "contents", "table of contents", "agenda",
}

// DoclingHybridChunker is intentionally a pass-through. Docling's official
// HybridChunker has already chosen semantic/token-aware boundaries, so running
// PageChunker (render + OCR) or another splitting algorithm would destroy
// those boundaries. Native parser JSON from formats outside Docling's verified
// allowlist is also accepted so a mixed dataset can use one downstream
// workflow without silently dropping its fallback documents.
type DoclingHybridChunker struct{}

func NewDoclingHybridChunker(_ map[string]any) (runtime.Component, error) {
	return &DoclingHybridChunker{}, nil
}

func (c *DoclingHybridChunker) Inputs() map[string]string  { return ChunkerInputs }
func (c *DoclingHybridChunker) Outputs() map[string]string { return ChunkerOutputs }

func (c *DoclingHybridChunker) Invoke(ctx context.Context, db *gorm.DB, inputs map[string]any) (map[string]any, error) {
	if inputs == nil {
		return inputErrorOutput(errors.New("DoclingHybridChunker received no input")), nil
	}
	upstream, err := decodeChunkerFromUpstream(inputs)
	if err != nil {
		return inputErrorOutput(err), nil
	}
	var chunks []schema.ChunkDoc
	switch upstream.OutputFormat {
	case schema.PayloadFormatChunks:
		chunks = upstream.Chunks
	case schema.PayloadFormatJSON, "":
		chunks = upstream.JSONResult
	case schema.PayloadFormatMarkdown:
		if upstream.MarkdownResult != nil && strings.TrimSpace(*upstream.MarkdownResult) != "" {
			chunks = []schema.ChunkDoc{{Text: *upstream.MarkdownResult, DocType: "text"}}
		}
	case schema.PayloadFormatText:
		if upstream.TextResult != nil && strings.TrimSpace(*upstream.TextResult) != "" {
			chunks = []schema.ChunkDoc{{Text: *upstream.TextResult, DocType: "text"}}
		}
	case schema.PayloadFormatHTML:
		if upstream.HTMLResult != nil && strings.TrimSpace(*upstream.HTMLResult) != "" {
			chunks = []schema.ChunkDoc{{Text: *upstream.HTMLResult, DocType: "text"}}
		}
	}
	if len(chunks) == 0 {
		return inputErrorOutput(errors.New("DoclingHybridChunker received zero chunks; refusing a silent successful ingestion")), nil
	}
	kbID, docID := resolveImageUploadContext(ctx, inputs)
	doclingChunks, previewSources := splitDoclingPresentationPreviewSources(chunks)
	presentationSemanticFallback := len(doclingChunks) == 0 && len(previewSources) > 0
	if len(doclingChunks) == 0 && len(previewSources) == 0 {
		return inputErrorOutput(errors.New("DoclingHybridChunker received no semantic chunks after removing presentation preview sources")), nil
	}
	if consumeNativePDFFallbackMarker(doclingChunks) {
		doclingChunks = attachPDFPagePreviews(ctx, db, upstream, doclingChunks)
	}
	if len(doclingChunks) > 0 {
		doclingChunks, err = expandDoclingPictureMedia(doclingChunks, docID)
		if err != nil {
			return inputErrorOutput(err), nil
		}
	}
	if len(previewSources) > 0 {
		previews, previewErr := buildDoclingSlidePreviews(ctx, previewSources, docID, presentationSemanticFallback)
		if previewErr != nil {
			return inputErrorOutput(previewErr), nil
		}
		if presentationSemanticFallback {
			doclingChunks = doclingPresentationSemanticFallback(previews, upstream.Name)
		}
		linkDoclingSemanticSlidePreviews(doclingChunks, previews, kbID)
		doclingChunks = append(doclingChunks, previews...)
	}
	return chunkOutputs(doclingChunks), nil
}

// consumeNativePDFFallbackMarker detects records emitted by the parser's
// empty-Docling-chunk recovery path and removes the private marker before the
// records reach the index. The caller then rebuilds them as page-scoped chunks
// with original-page previews and local OCR.
func consumeNativePDFFallbackMarker(chunks []schema.ChunkDoc) bool {
	fallback := false
	for index := range chunks {
		raw, ok := chunks[index].Extra["docling_native_pdf_fallback"]
		if !ok {
			continue
		}
		var marked bool
		if json.Unmarshal(raw, &marked) == nil && marked {
			fallback = true
		}
		chunks[index].Extra = cloneChunkExtra(chunks[index].Extra)
		delete(chunks[index].Extra, "docling_native_pdf_fallback")
	}
	return fallback
}

// doclingPresentationSemanticFallback turns OCR-enriched full-slide previews
// into ordinary semantic chunks when both Docling and the native Office text
// extractor find no text boundaries. Preview bytes remain in their dedicated
// chunks; these clones carry only text/page identity and are therefore safe to
// contextualize and embed. A bounded filename/page label is retained when OCR
// cannot recover text so an image-only slide is never silently discarded.
func doclingPresentationSemanticFallback(previews []schema.ChunkDoc, filename string) []schema.ChunkDoc {
	semantic := make([]schema.ChunkDoc, 0, len(previews))
	filename = strings.TrimSpace(filename)
	for index := range previews {
		preview := previews[index]
		page, ok := presentationPageNumber(preview)
		if !ok {
			page = index + 1
		}
		text := strings.TrimSpace(preview.Text)
		if text == "" || text == fmt.Sprintf("[Page %d]", page) {
			if filename == "" {
				filename = "presentation"
			}
			text = fmt.Sprintf("%s — slide %d visual content", filename, page)
		}
		chunk := preview
		chunk.ID = ""
		chunk.Text = text
		chunk.DocType = "text"
		chunk.Image, chunk.ImgID, chunk.MediaRef = "", "", nil
		chunk.Extra = cloneChunkExtra(preview.Extra)
		for _, key := range []string{
			doclingSlidePreviewKey,
			doclingImageRoleKey,
			doclingPreviewAutoDisplayKey,
			doclingPreviewReasonKey,
		} {
			delete(chunk.Extra, key)
		}
		setChunkExtraJSON(&chunk, "docling_native_semantic_fallback", true)
		setChunkExtraJSON(&chunk, "docling_page_numbers", []int{page})
		semantic = append(semantic, chunk)
	}
	return semantic
}

// linkDoclingSemanticSlidePreviews attaches the most relevant full-slide
// preview to every semantic text chunk. Retrieval ranks semantic chunks above
// the derived image-only chunks; without this link the answer has correct text
// but no image_id, which prevents the UI from showing the supporting slide.
//
// Docling records every page touched by a hybrid chunk. We restrict candidates
// to those pages, discard cover/TOC previews, then choose the slide with the
// greatest lexical overlap. The semantic chunk references the preview's
// deterministic object ID, so bytes are uploaded once by the preview chunk and
// shared by both search results.
func linkDoclingSemanticSlidePreviews(chunks, previews []schema.ChunkDoc, kbID string) {
	if strings.TrimSpace(kbID) == "" || len(chunks) == 0 || len(previews) == 0 {
		return
	}
	byPage := make(map[int]*schema.ChunkDoc, len(previews))
	for index := range previews {
		preview := &previews[index]
		page, ok := presentationPageNumber(*preview)
		if !ok || !doclingPreviewAutoDisplay(*preview) || strings.TrimSpace(preview.ID) == "" {
			continue
		}
		byPage[page] = preview
	}
	for index := range chunks {
		chunk := &chunks[index]
		if chunk.ImgID != "" || chunk.Image != "" || chunk.MediaRef != nil || strings.TrimSpace(chunk.Text) == "" {
			continue
		}
		var selected *schema.ChunkDoc
		selectedPage, selectedScore := 0, -1
		for _, page := range doclingChunkPageNumbers(*chunk) {
			preview := byPage[page]
			if preview == nil {
				continue
			}
			score := doclingTextOverlapScore(chunk.Text, preview.Text)
			if score > selectedScore {
				selected, selectedPage, selectedScore = preview, page, score
			}
		}
		if selected == nil {
			continue
		}
		chunk.ImgID = kbID + "-" + selected.ID
		setChunkExtraJSON(chunk, "docling_linked_slide_page", selectedPage)
		setChunkExtraJSON(chunk, "docling_linked_slide_preview", true)
	}
}

func doclingPreviewAutoDisplay(preview schema.ChunkDoc) bool {
	if raw, ok := preview.Extra[doclingPreviewAutoDisplayKey]; ok {
		var value bool
		if json.Unmarshal(raw, &value) == nil {
			return value
		}
	}
	visible, _ := doclingPreviewDisplayPolicy(preview)
	return visible
}

func doclingChunkPageNumbers(chunk schema.ChunkDoc) []int {
	var pages []int
	if raw, ok := chunk.Extra["docling_page_numbers"]; ok {
		_ = json.Unmarshal(raw, &pages)
	}
	if len(pages) == 0 && chunk.PageNumber != nil && *chunk.PageNumber > 0 {
		pages = append(pages, *chunk.PageNumber)
	}
	seen := make(map[int]struct{}, len(pages))
	out := make([]int, 0, len(pages))
	for _, page := range pages {
		if page <= 0 {
			continue
		}
		if _, duplicate := seen[page]; duplicate {
			continue
		}
		seen[page] = struct{}{}
		out = append(out, page)
	}
	return out
}

func doclingTextOverlapScore(left, right string) int {
	leftTokens := doclingOverlapTokens(left)
	rightTokens := doclingOverlapTokens(right)
	if len(leftTokens) == 0 || len(rightTokens) == 0 {
		return 0
	}
	score := 0
	for token := range rightTokens {
		if _, ok := leftTokens[token]; ok {
			score++
		}
	}
	return score
}

func doclingOverlapTokens(text string) map[string]struct{} {
	tokens := make(map[string]struct{})
	fields := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !(r >= '0' && r <= '9') && !(r >= 'a' && r <= 'z') && !(r >= '\uac00' && r <= '\ud7a3')
	})
	for _, field := range fields {
		if len([]rune(field)) >= 2 {
			tokens[field] = struct{}{}
		}
	}
	return tokens
}

// splitDoclingPresentationPreviewSources removes native slide geometry/media
// records from the ordinary Docling stream. They are an internal rendering
// input, not indexable chunks, and must never leak into retrieval as duplicate
// text or detached individual shapes.
func splitDoclingPresentationPreviewSources(chunks []schema.ChunkDoc) ([]schema.ChunkDoc, []schema.ChunkDoc) {
	docling := make([]schema.ChunkDoc, 0, len(chunks))
	preview := make([]schema.ChunkDoc, 0)
	for _, chunk := range chunks {
		marked := false
		if raw, ok := chunk.Extra[doclingPresentationPreviewSourceKey]; ok {
			_ = json.Unmarshal(raw, &marked)
		}
		if !marked {
			docling = append(docling, chunk)
			continue
		}
		chunk.Extra = cloneChunkExtra(chunk.Extra)
		delete(chunk.Extra, doclingPresentationPreviewSourceKey)
		preview = append(preview, chunk)
	}
	return docling, preview
}

// buildDoclingSlidePreviews reuses the official Presentation chunker's
// renderer to compose one bounded full-slide PNG from every group of native
// PPTX shape records. The slide preview is a separate retrieval chunk with a
// deterministic media identity and exact page metadata; Docling semantic text
// chunks remain untouched.
func buildDoclingSlidePreviews(ctx context.Context, sources []schema.ChunkDoc, docID string, enableOCR bool) ([]schema.ChunkDoc, error) {
	previews := mergePresentationPages(ctx, sources, enableOCR)
	if len(previews) == 0 {
		return nil, errors.New("DoclingHybridChunker presentation preview rendering produced zero slides")
	}
	for index := range previews {
		preview := &previews[index]
		if preview.MediaRef == nil && strings.TrimSpace(preview.Image) == "" {
			return nil, fmt.Errorf("DoclingHybridChunker slide %d preview has no image", index+1)
		}
		if preview.Extra == nil {
			preview.Extra = make(map[string]json.RawMessage)
		}
		setChunkExtraJSON(preview, doclingSlidePreviewKey, true)
		setChunkExtraJSON(preview, doclingImageRoleKey, doclingImageRoleFullSlide)
		autoDisplay, reason := doclingPreviewDisplayPolicy(*preview)
		setChunkExtraJSON(preview, doclingPreviewAutoDisplayKey, autoDisplay)
		setChunkExtraJSON(preview, doclingPreviewReasonKey, reason)
		page, ok := presentationPageNumber(*preview)
		if !ok {
			page = index + 1
		}
		// Persist an explicit full-slide position so the indexed chunk retains
		// its slide number. PageNumber is pipeline-only metadata and is removed at
		// the storage boundary; positions is the canonical carrier from which
		// page_num_int / position_int are derived.
		if positions, err := json.Marshal([][]float64{{float64(page), 0, 1, 0, 1}}); err == nil {
			preview.Positions = positions
		}
		preview.ID = common.ChunkID(docID, fmt.Sprintf("%s:%d", doclingImageRoleFullSlide, page))
	}
	return previews, nil
}

func setChunkExtraJSON(chunk *schema.ChunkDoc, key string, value any) {
	encoded, err := json.Marshal(value)
	if err == nil {
		chunk.Extra[key] = encoded
	}
}

// doclingPreviewDisplayPolicy lets the answer UI suppress predictable cover
// and table-of-contents screenshots without deleting them from provenance.
// The image remains downloadable/citable; only automatic inline display is
// discouraged. Content slides default to visible.
func doclingPreviewDisplayPolicy(preview schema.ChunkDoc) (bool, string) {
	text := strings.ToLower(strings.TrimSpace(preview.Text))
	for _, marker := range doclingTOCMarkers {
		if strings.Contains(text, marker) {
			return false, doclingPreviewReasonTOC
		}
	}
	page, _ := presentationPageNumber(preview)
	// A manual deck's first slide is provenance, not answer evidence. Keep it
	// searchable/downloadable, but never insert it automatically into answers.
	if page == 1 {
		return false, doclingPreviewReasonCover
	}
	return true, doclingPreviewReasonContent
}

type doclingPictureMedia struct {
	Ref          string          `json:"ref"`
	Image        string          `json:"image"`
	Positions    json.RawMessage `json:"positions"`
	Descriptions []string        `json:"descriptions"`
}

// expandDoclingPictureMedia bridges Docling's many-doc_items-per-hybrid-chunk
// model to RAGFlow's one-img_id-per-chunk index schema. The first media chunk
// keeps the official HybridChunker text while aggregate source provenance is
// retained separately;
// subsequent pictures become linked media chunks whose text is their native
// picture description. Every output owns one image payload, so the existing
// upload decorator creates one durable img_id per picture and removes bytes
// before indexing.
func expandDoclingPictureMedia(chunks []schema.ChunkDoc, docID string) ([]schema.ChunkDoc, error) {
	out := make([]schema.ChunkDoc, 0, len(chunks))
	for _, source := range chunks {
		raw, exists := source.Extra["docling_picture_media"]
		if !exists {
			out = append(out, source)
			continue
		}
		var media []doclingPictureMedia
		if err := json.Unmarshal(raw, &media); err != nil {
			return nil, fmt.Errorf("DoclingHybridChunker decode picture media: %w", err)
		}
		if len(media) == 0 {
			return nil, errors.New("DoclingHybridChunker received empty docling_picture_media")
		}
		if len(media) > 1 {
			parent := source
			parent.Extra = cloneChunkExtra(source.Extra)
			delete(parent.Extra, "docling_picture_media")
			parent.Image = ""
			parent.DocType = "text"
			out = append(out, parent)
		}
		for index, picture := range media {
			if strings.TrimSpace(picture.Image) == "" {
				return nil, fmt.Errorf("DoclingHybridChunker picture %d has no image payload", index)
			}
			chunk := source
			chunk.Extra = cloneChunkExtra(source.Extra)
			delete(chunk.Extra, "docling_picture_media")
			delete(chunk.Extra, "docling_picture_refs")
			delete(chunk.Extra, "docling_picture_descriptions")
			chunk.Image = picture.Image
			chunk.DocType = "image"
			if encoded, err := json.Marshal(picture.Ref); err == nil {
				chunk.Extra["docling_picture_ref"] = encoded
			}
			if encoded, err := json.Marshal(index); err == nil {
				chunk.Extra["docling_media_ordinal"] = encoded
			}
			if encoded, err := json.Marshal(picture.Descriptions); err == nil {
				chunk.Extra["docling_picture_descriptions"] = encoded
			}
			if parent, ok := chunk.Extra["docling_chunk_index"]; ok {
				chunk.Extra["docling_parent_chunk_index"] = append(json.RawMessage(nil), parent...)
			}
			if len(source.Positions) > 0 {
				chunk.Extra["docling_source_positions"] = append(json.RawMessage(nil), source.Positions...)
			}
			if len(picture.Positions) > 0 && string(picture.Positions) != "null" {
				chunk.Positions = append(json.RawMessage(nil), picture.Positions...)
				chunk.PDFPositions = append(json.RawMessage(nil), picture.Positions...)
			}
			if len(media) > 1 {
				description := strings.TrimSpace(strings.Join(picture.Descriptions, "\n"))
				if description == "" {
					description = fmt.Sprintf("[Picture %d/%d]", index+1, len(media))
				}
				chunk.Text = description
			}
			identity, identityErr := doclingPictureIdentity(chunk.Text, picture, index)
			if identityErr != nil {
				return nil, fmt.Errorf("DoclingHybridChunker picture %d identity: %w", index, identityErr)
			}
			chunk.ID = common.ChunkID(docID, identity)
			out = append(out, chunk)
		}
	}
	if len(out) == 0 {
		return nil, errors.New("DoclingHybridChunker picture expansion produced zero chunks")
	}
	return out, nil
}

// doclingPictureIdentity makes every Docling picture a distinct durable media
// object even when two figures have the same description and no geometry.
// Ref and ordinal preserve source identity; the decoded payload digest keeps
// the identity content-addressed across equivalent data-URL wrappers. The
// body remains part of the digest so changed descriptions invalidate caches.
func doclingPictureIdentity(text string, picture doclingPictureMedia, ordinal int) (string, error) {
	payload, err := decodeChunkImage(picture.Image)
	if err != nil {
		return "", fmt.Errorf("decode image payload: %w", err)
	}
	payloadDigest := sha256.Sum256(payload)
	canonical, err := json.Marshal(struct {
		Text          string `json:"text"`
		Ref           string `json:"ref"`
		Ordinal       int    `json:"ordinal"`
		PayloadSHA256 string `json:"payload_sha256"`
	}{
		Text:          text,
		Ref:           strings.TrimSpace(picture.Ref),
		Ordinal:       ordinal,
		PayloadSHA256: hex.EncodeToString(payloadDigest[:]),
	})
	if err != nil {
		return "", err
	}
	identityDigest := sha256.Sum256(canonical)
	return mediaChunkIDPrefix + "docling:" + hex.EncodeToString(identityDigest[:]), nil
}

func cloneChunkExtra(source map[string]json.RawMessage) map[string]json.RawMessage {
	cloned := make(map[string]json.RawMessage, len(source))
	for key, raw := range source {
		cloned[key] = append(json.RawMessage(nil), raw...)
	}
	return cloned
}

func init() {
	MustRegisterChunker(ComponentNameDoclingHybridChunker)
}
