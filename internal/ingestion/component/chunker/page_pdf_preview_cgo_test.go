//go:build cgo

package chunker

import (
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"strings"
	"testing"

	deepdocpdf "ragflow/internal/deepdoc/parser/pdf"
	"ragflow/internal/ingestion/component/schema"
)

func TestAttachPDFPagePreviewsRendersEveryPage(t *testing.T) {
	engine := &deepdocpdf.MockEngine{
		NumPages: 2,
		RenderPageImageFunc: func(page int, _ float64) (image.Image, error) {
			img := image.NewRGBA(image.Rect(0, 0, 320, 180))
			fill := color.RGBA{R: uint8(40 + page*100), A: 255}
			for y := 0; y < img.Bounds().Dy(); y++ {
				for x := 0; x < img.Bounds().Dx(); x++ {
					img.Set(x, y, fill)
				}
			}
			return img, nil
		},
	}
	page := 1
	chunks := attachPDFPagePreviewsWithEngine(t.Context(), engine, []schema.ChunkDoc{{Text: "first", PageNumber: &page}})
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want every PDF page", len(chunks))
	}
	for i, chunk := range chunks {
		if chunk.PageNumber == nil || *chunk.PageNumber != i+1 {
			t.Fatalf("chunk %d page = %v", i, chunk.PageNumber)
		}
		if chunk.DocType != "image" || !strings.HasPrefix(chunk.Image, "data:image/png;base64,") {
			t.Fatalf("chunk %d preview missing", i)
		}
	}
	if chunks[1].Text != "[Page 2]" {
		t.Fatalf("empty PDF page fallback text = %q", chunks[1].Text)
	}
	var firstPositions, secondPositions [][]any
	if err := json.Unmarshal(chunks[0].Positions, &firstPositions); err != nil {
		t.Fatalf("first positions: %v", err)
	}
	if err := json.Unmarshal(chunks[1].Positions, &secondPositions); err != nil {
		t.Fatalf("second positions: %v", err)
	}
	if got := firstPositions[0][0]; got != float64(0) {
		t.Fatalf("page 1 position index = %v, want 0", got)
	}
	if got := secondPositions[0][0]; got != float64(1) {
		t.Fatalf("page 2 position index = %v, want 1", got)
	}
}

func TestAttachPDFPagePreviewsPreservesParserPositions(t *testing.T) {
	engine := &deepdocpdf.MockEngine{
		NumPages: 1,
		RenderPageImageFunc: func(int, float64) (image.Image, error) {
			return image.NewRGBA(image.Rect(0, 0, 20, 10)), nil
		},
	}
	page := 1
	originalPositions := json.RawMessage(`[[0,11,12,13,14]]`)
	originalPDFPositions := json.RawMessage(`[[0,21,22,23,24]]`)
	chunks := attachPDFPagePreviewsWithEngine(t.Context(), engine, []schema.ChunkDoc{{
		Text:         "searchable",
		PageNumber:   &page,
		Positions:    originalPositions,
		PDFPositions: originalPDFPositions,
	}})
	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(chunks))
	}
	if string(chunks[0].Positions) != string(originalPositions) || string(chunks[0].PDFPositions) != string(originalPDFPositions) {
		t.Fatalf("parser positions were replaced: positions=%s pdf_positions=%s", chunks[0].Positions, chunks[0].PDFPositions)
	}
}

func TestAttachPDFPagePreviewsRenderFailureKeepsTextChunk(t *testing.T) {
	engine := &deepdocpdf.MockEngine{
		NumPages: 1,
		RenderPageImageFunc: func(int, float64) (image.Image, error) {
			return nil, errors.New("render unavailable")
		},
	}
	page := 1
	chunks := attachPDFPagePreviewsWithEngine(t.Context(), engine, []schema.ChunkDoc{{Text: "searchable", DocType: "text", PageNumber: &page}})
	if len(chunks) != 1 || chunks[0].Text != "searchable" || chunks[0].Image != "" || chunks[0].MediaRef != nil {
		t.Fatalf("render failure did not degrade to searchable text: %#v", chunks)
	}
}
