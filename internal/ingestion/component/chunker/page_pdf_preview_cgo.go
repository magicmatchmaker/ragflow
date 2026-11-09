//go:build cgo

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
	"encoding/json"
	"fmt"

	"go.uber.org/zap"
	"gorm.io/gorm"

	"ragflow/internal/common"
	deepdocpdf "ragflow/internal/deepdoc/parser/pdf"
	deepdoctype "ragflow/internal/deepdoc/parser/type"
	"ragflow/internal/ingestion/component/schema"
)

const maxPresentationPDFPages = 10_000

func attachPDFPagePreviews(ctx context.Context, db *gorm.DB, upstream schema.ChunkerFromUpstream, chunks []schema.ChunkDoc) []schema.ChunkDoc {
	engine, err := newPDFEngineFromUpstream(ctx, db, upstream)
	if err != nil {
		common.Warn("PageChunker: PDF preview source could not be opened; retaining searchable text", zap.Error(err))
		return chunks
	}
	if engine == nil {
		return chunks
	}
	defer engine.Close()
	return attachPDFPagePreviewsWithEngine(ctx, engine, chunks)
}

func attachPDFPagePreviewsWithEngine(ctx context.Context, engine deepdoctype.PDFEngine, chunks []schema.ChunkDoc) []schema.ChunkDoc {
	pageCount, err := engine.PageCount()
	if err != nil || pageCount <= 0 || pageCount > maxPresentationPDFPages {
		common.Warn("PageChunker: PDF page count unavailable or outside preview limit",
			zap.Int("page_count", pageCount), zap.Error(err))
		return chunks
	}

	byPage := make(map[int]schema.ChunkDoc, len(chunks))
	var unidentified []schema.ChunkDoc
	for _, chunk := range chunks {
		page, ok := presentationPageNumber(chunk)
		if !ok || page > pageCount {
			unidentified = append(unidentified, chunk)
			continue
		}
		byPage[page] = chunk
	}

	out := make([]schema.ChunkDoc, 0, pageCount+len(unidentified))
	for page := 1; page <= pageCount; page++ {
		chunk, ok := byPage[page]
		if !ok {
			chunk = schema.ChunkDoc{Text: fmt.Sprintf("[Page %d]", page), DocType: "text"}
			chunk.PageNumber = intPtr(page)
		}
		// PDF positions use a zero-based page index. The web client adds one
		// when presenting the page number, matching parser-produced positions.
		position, _ := json.Marshal([][]any{{page - 1, 0, 0, 0, 0}})
		if len(chunk.Positions) == 0 {
			chunk.Positions = position
		}
		if len(chunk.PDFPositions) == 0 {
			chunk.PDFPositions = position
		}

		img, renderErr := deepdocpdf.RenderPageToImage(engine, page-1)
		if renderErr != nil || img == nil {
			common.Warn("PageChunker: PDF page preview render failed; retaining searchable text",
				zap.Int("page", page), zap.Error(renderErr))
			out = append(out, chunk)
			continue
		}
		preview, encodeErr := encodeBoundedPresentationImage(img)
		if encodeErr != nil {
			common.Warn("PageChunker: PDF page preview encoding failed; retaining searchable text",
				zap.Int("page", page), zap.Error(encodeErr))
			out = append(out, chunk)
			continue
		}
		stored, storeErr := storePresentationPreview(ctx, preview)
		if storeErr != nil {
			common.Warn("PageChunker: PDF page preview cache write failed; retaining searchable text",
				zap.Int("page", page), zap.Error(storeErr))
			out = append(out, chunk)
			continue
		}
		if stored != nil {
			chunk.Image, chunk.ImgID, chunk.MediaRef = stored.Image, "", stored.MediaRef
			chunk.DocType = "image"
		}
		out = append(out, chunk)
	}
	return append(out, unidentified...)
}
