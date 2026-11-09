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
	"strings"

	"go.uber.org/zap"
	"gorm.io/gorm"

	"ragflow/internal/common"
	deepdocpdf "ragflow/internal/deepdoc/parser/pdf"
	deepdoctype "ragflow/internal/deepdoc/parser/type"
	"ragflow/internal/ingestion/component/schema"
	parserlib "ragflow/internal/parser/parser"
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
		if !ok || page < 1 || page > pageCount {
			unidentified = append(unidentified, chunk)
			continue
		}
		if existing, duplicate := byPage[page]; duplicate {
			byPage[page] = mergePDFPageChunk(existing, chunk)
		} else {
			byPage[page] = chunk
		}
	}
	if len(unidentified) > 0 {
		// A DeepDOC result can contain a document-level or otherwise unpositioned
		// text record. PageChunker promises exactly one output per physical PDF
		// page, so never append that record as a 53rd pseudo-page. Merge it into
		// page one instead, preserving searchable text without inventing page 53.
		anchor, ok := byPage[1]
		if !ok {
			anchor = schema.ChunkDoc{Text: "[Page 1]", DocType: "text", PageNumber: intPtr(1)}
		}
		for _, chunk := range unidentified {
			anchor = mergePDFPageChunk(anchor, chunk)
		}
		byPage[1] = anchor
		common.Warn("PageChunker: merged unpositioned PDF records into page one",
			zap.Int("record_count", len(unidentified)))
	}

	out := make([]schema.ChunkDoc, 0, pageCount)
	var ocrAnalyzer deepdoctype.DocAnalyzer
	var ocrAnalyzerErr error
	ocrAnalyzerResolved := false
	for page := 1; page <= pageCount; page++ {
		chunk, ok := byPage[page]
		if !ok {
			chunk = schema.ChunkDoc{Text: fmt.Sprintf("[Page %d]", page), DocType: "text"}
		}
		chunk.PageNumber = intPtr(page)
		chunk.Positions, chunk.PDFPositions = normalizedPDFPagePositions(chunk, page-1)

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
		if !ocrAnalyzerResolved {
			ocrAnalyzer, ocrAnalyzerErr = resolvePresentationOCRAnalyzer()
			ocrAnalyzerResolved = true
			if ocrAnalyzerErr != nil {
				common.Warn("PageChunker: local PDF page OCR is unavailable; retaining parser text and preview", zap.Error(ocrAnalyzerErr))
			}
		}
		if ocrAnalyzer != nil {
			ocrText, ocrErr := extractPresentationOCR(ctx, preview, ocrAnalyzer)
			if ocrErr != nil {
				common.Warn("PageChunker: local PDF page OCR failed; retaining parser text and preview",
					zap.Int("page", page), zap.Error(ocrErr))
			} else {
				chunk.Text = mergePresentationOCR(chunk.Text, ocrText)
			}
		}
		if strings.TrimSpace(chunk.Text) == "" {
			chunk.Text = fmt.Sprintf("[Page %d]", page)
		}
		out = append(out, chunk)
	}
	return out
}

func mergePDFPageChunk(dst, src schema.ChunkDoc) schema.ChunkDoc {
	extra := strings.TrimSpace(src.Text)
	if extra != "" {
		existing := strings.TrimSpace(dst.Text)
		existingKey, extraKey := normalizePresentationOCR(existing), normalizePresentationOCR(extra)
		switch {
		case existing == "":
			dst.Text = extra
		case extraKey != "" && extraKey != existingKey && !strings.Contains(existingKey, extraKey):
			dst.Text = existing + "\n\n" + extra
		}
	}
	if dst.Image == "" && dst.ImgID == "" && dst.MediaRef == nil {
		dst.Image, dst.ImgID, dst.MediaRef = src.Image, src.ImgID, src.MediaRef
	}
	if len(dst.Positions) == 0 && len(src.Positions) > 0 {
		dst.Positions = append(json.RawMessage(nil), src.Positions...)
	}
	if len(dst.PDFPositions) == 0 && len(src.PDFPositions) > 0 {
		dst.PDFPositions = append(json.RawMessage(nil), src.PDFPositions...)
	}
	return dst
}

func normalizedPDFPagePositions(chunk schema.ChunkDoc, pageIndex int) (json.RawMessage, json.RawMessage) {
	normalize := func(raw json.RawMessage) json.RawMessage {
		rows := parserlib.PDFPositionRowsFromJSON(raw)
		if len(rows) == 0 {
			return nil
		}
		normalized := make([][]any, 0, len(rows))
		for _, row := range rows {
			normalized = append(normalized, []any{pageIndex, row[1], row[2], row[3], row[4]})
		}
		encoded, err := json.Marshal(normalized)
		if err != nil {
			return nil
		}
		return encoded
	}
	positions := normalize(chunk.Positions)
	pdfPositions := normalize(chunk.PDFPositions)
	if len(positions) == 0 {
		positions = append(json.RawMessage(nil), pdfPositions...)
	}
	if len(pdfPositions) == 0 {
		pdfPositions = append(json.RawMessage(nil), positions...)
	}
	if len(positions) == 0 {
		fallback, _ := json.Marshal([][]any{{pageIndex, 0, 0, 0, 0}})
		positions = fallback
		pdfPositions = append(json.RawMessage(nil), fallback...)
	}
	return positions, pdfPositions
}
