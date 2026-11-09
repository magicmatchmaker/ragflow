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
	originalPositions := json.RawMessage(`[[1,11,12,13,14]]`)
	originalPDFPositions := json.RawMessage(`[[1,21,22,23,24]]`)
	chunks := attachPDFPagePreviewsWithEngine(t.Context(), engine, []schema.ChunkDoc{{
		Text:         "searchable",
		PageNumber:   &page,
		Positions:    originalPositions,
		PDFPositions: originalPDFPositions,
	}})
	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(chunks))
	}
	var positions, pdfPositions [][]float64
	if err := json.Unmarshal(chunks[0].Positions, &positions); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(chunks[0].PDFPositions, &pdfPositions); err != nil {
		t.Fatal(err)
	}
	if len(positions) != 1 || len(pdfPositions) != 1 ||
		positions[0][0] != 0 || pdfPositions[0][0] != 0 ||
		positions[0][1] != 11 || positions[0][4] != 14 ||
		pdfPositions[0][1] != 21 || pdfPositions[0][4] != 24 {
		t.Fatalf("positions not normalized with geometry preserved: positions=%v pdf_positions=%v", positions, pdfPositions)
	}
}

func TestAttachPDFPagePreviewsMergesUnidentifiedWithoutExtraChunk(t *testing.T) {
	usePresentationOCRAnalyzer(t, nil, false)
	engine := &deepdocpdf.MockEngine{
		NumPages: 2,
		RenderPageImageFunc: func(int, float64) (image.Image, error) {
			return image.NewRGBA(image.Rect(0, 0, 20, 10)), nil
		},
	}
	page1, page2 := 1, 2
	chunks := attachPDFPagePreviewsWithEngine(t.Context(), engine, []schema.ChunkDoc{
		{Text: "page one", PageNumber: &page1},
		{Text: "page two", PageNumber: &page2, Positions: json.RawMessage(`[[2,5,6,7,8]]`)},
		{Text: "unpositioned document text"},
		{Text: "page one", PageNumber: &page1, Positions: json.RawMessage(`[[1,1,2,3,4]]`)},
	})
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want exactly one per PDF page", len(chunks))
	}
	if !strings.Contains(chunks[0].Text, "unpositioned document text") || strings.Count(chunks[0].Text, "page one") != 1 {
		t.Fatalf("unidentified/deduplicated text merge failed: %q", chunks[0].Text)
	}
	for i, chunk := range chunks {
		var positions, pdfPositions [][]float64
		if err := json.Unmarshal(chunk.Positions, &positions); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(chunk.PDFPositions, &pdfPositions); err != nil {
			t.Fatal(err)
		}
		if len(positions) == 0 || len(pdfPositions) == 0 || positions[0][0] != float64(i) || pdfPositions[0][0] != float64(i) {
			t.Fatalf("page %d positions = %v / %v, want zero-based index %d", i+1, positions, pdfPositions, i)
		}
		if i == 0 && (positions[0][1] != 1 || positions[0][4] != 4) {
			t.Fatalf("duplicate page merge lost source geometry: %v", positions)
		}
	}
}

func TestAttachPDFPagePreviewsAddsFullPageOCRText(t *testing.T) {
	const recognized = "synthetic PDF screenshot marker"
	usePresentationOCRAnalyzer(t, presentationOCRAnalyzer(recognized), true)
	engine := &deepdocpdf.MockEngine{
		NumPages: 1,
		RenderPageImageFunc: func(int, float64) (image.Image, error) {
			return image.NewRGBA(image.Rect(0, 0, 100, 60)), nil
		},
	}
	page := 1
	chunks := attachPDFPagePreviewsWithEngine(t.Context(), engine, []schema.ChunkDoc{{Text: "parser text", PageNumber: &page}})
	if len(chunks) != 1 || !strings.Contains(chunks[0].Text, recognized) || !strings.Contains(chunks[0].Text, "parser text") {
		t.Fatalf("full-page OCR did not merge with parser text: %#v", chunks)
	}
	if chunks[0].Image == "" && chunks[0].MediaRef == nil {
		t.Fatal("full-page preview was lost while adding OCR")
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
