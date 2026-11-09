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

package chunker

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"strings"
	"testing"

	"ragflow/internal/entity"
	"ragflow/internal/ingestion/component/schema"
	"ragflow/internal/parser/parser"
)

func presentationTestImage(t *testing.T, fill color.Color) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 8, 8))
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			img.Set(x, y, fill)
		}
	}
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, img); err != nil {
		t.Fatal(err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(encoded.Bytes())
}

func decodePresentationTestPreview(t *testing.T, value any) image.Image {
	t.Helper()
	encoded, ok := value.(string)
	if !ok {
		t.Fatalf("preview type = %T", value)
	}
	data, err := decodeChunkImage(encoded)
	if err != nil {
		t.Fatal(err)
	}
	img, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	return img
}

// slideChunksOf drives PageChunker.Invoke and returns the emitted maps.
func slideChunksOf(t *testing.T, inputs map[string]any) []map[string]any {
	t.Helper()
	comp, err := NewPageChunker(nil)
	if err != nil {
		t.Fatalf("NewPageChunker: %v", err)
	}
	out, err := comp.Invoke(context.Background(), nil, inputs)
	if err != nil {
		t.Fatalf("PageChunker.Invoke: %v", err)
	}
	chunks, ok := schema.ChunkCarrierToMaps(out["chunks"])
	if !ok {
		t.Fatalf("chunks not []map[string]any: %T", out["chunks"])
	}
	return chunks
}

// TestPageChunker_OneChunkPerSlide ports rag/app/presentation.py's
// contract: each slide becomes exactly one chunk, and per-slide metadata
// (page_number, image) survives the chunker round-trip.
func TestPageChunker_OneChunkPerSlide(t *testing.T) {
	red := presentationTestImage(t, color.RGBA{R: 255, A: 255})
	blue := presentationTestImage(t, color.RGBA{B: 255, A: 255})
	chunks := slideChunksOf(t, map[string]any{
		"name":          "deck.pptx",
		"output_format": "json",
		"json": []map[string]any{
			{
				"text":         "Slide 1 content",
				"doc_type_kwd": "image",
				"page_number":  float64(1),
				"image":        red,
			},
			{
				"text":         "Slide 2 content",
				"doc_type_kwd": "image",
				"page_number":  float64(2),
				"image":        blue,
			},
			{
				"text":         "Slide 3 content",
				"doc_type_kwd": "image",
				"page_number":  float64(3),
			},
		},
	})

	if len(chunks) != 3 {
		t.Fatalf("expected 3 slides, got %d", len(chunks))
	}

	for i, ck := range chunks {
		wantPage := float64(i + 1)
		if ck["page_number"] != wantPage {
			t.Errorf("slide %d: page_number = %v, want %v", i+1, ck["page_number"], wantPage)
		}
		if ck["doc_type_kwd"] != "image" {
			t.Errorf("slide %d: doc_type_kwd = %v, want image", i+1, ck["doc_type_kwd"])
		}
		if ck["text"] == "" {
			t.Errorf("slide %d: empty text", i+1)
		}
	}

	// Per-slide image attachments must be preserved (the key win over
	// TokenChunker "one", which drops media context).
	if image, _ := chunks[0]["image"].(string); !strings.HasPrefix(image, "data:image/png;base64,") {
		t.Errorf("slide 1: composite preview missing")
	}
	if image, _ := chunks[1]["image"].(string); !strings.HasPrefix(image, "data:image/png;base64,") {
		t.Errorf("slide 2: composite preview missing")
	}
	// A text-only slide still receives a full-slide preview so citations have
	// the same visual contract as image-heavy slides.
	if image, _ := chunks[2]["image"].(string); !strings.HasPrefix(image, "data:image/png;base64,") {
		t.Errorf("slide 3: text-only preview missing")
	}
}

// TestPageChunker_Empty yields no chunks when there is no data.
func TestPageChunker_Empty(t *testing.T) {
	chunks := slideChunksOf(t, map[string]any{
		"name":          "deck.pptx",
		"output_format": "json",
		"json":          []map[string]any{},
	})
	if len(chunks) != 0 {
		t.Fatalf("expected 0 chunks for empty input, got %d", len(chunks))
	}
}

// TestPageChunker_NilInput yields no chunks.
func TestPageChunker_NilInput(t *testing.T) {
	chunks := slideChunksOf(t, nil)
	if len(chunks) != 0 {
		t.Fatalf("expected 0 chunks for nil input, got %d", len(chunks))
	}
}

func TestPageChunkerMergesSlideTableItems(t *testing.T) {
	table := map[string]any{"grid": []any{[]any{map[string]any{"text": "cell"}}}, "header_rows": 1}
	one := presentationTestImage(t, color.RGBA{R: 255, A: 255})
	two := presentationTestImage(t, color.RGBA{B: 255, A: 255})
	chunks := slideChunksOf(t, map[string]any{"name": "deck.pptx", "output_format": "json", "json": []map[string]any{
		{"text": "Before", "doc_type_kwd": "text", "slide_number": 1},
		{"doc_type_kwd": "table", "slide_number": 1, "source_table_id": "table-1", "table": table},
		{"doc_type_kwd": "image", "slide_number": 1, "parent_table_id": "table-1", "image": one},
		{"doc_type_kwd": "table", "slide_number": 1, "source_table_id": "table-2", "table": table},
		{"doc_type_kwd": "image", "slide_number": 1, "parent_table_id": "table-2", "image": two},
		{"text": "After", "doc_type_kwd": "text", "slide_number": 1},
	}})
	if len(chunks) != 1 {
		t.Fatalf("expected one chunk for one slide: %#v", chunks)
	}
	text, _ := chunks[0]["text"].(string)
	if !strings.Contains(text, "Before") || !strings.Contains(text, "cell") || !strings.Contains(text, "After") {
		t.Fatalf("searchable slide text lost: %#v", chunks)
	}
	if preview, _ := chunks[0]["image"].(string); chunks[0]["doc_type_kwd"] != "image" || !strings.HasPrefix(preview, "data:image/png;base64,") {
		t.Fatalf("composite slide media lost: %#v", chunks)
	}
	if got := fmt.Sprint(chunks[0]["slide_number"]); got != "1" {
		t.Fatalf("slide_number lost: %#v", chunks[0])
	}
}

// TestPageChunkerPreservesTypedSlideTableItems covers the in-memory
// []parser.DocItem leg, which bypasses the JSON-map decode path.
func TestPageChunkerMergesTypedSlideTableItems(t *testing.T) {
	table := &entity.TableData{
		Grid:       [][]entity.TableCell{{{Text: "cell"}}},
		HeaderRows: 1,
	}
	maps := []map[string]any{
		{"text": "Before", "doc_type_kwd": "text", "slide_number": 1},
		{"doc_type_kwd": "table", "slide_number": 1, "source_table_id": "table-1", "table": table},
		{"doc_type_kwd": "image", "slide_number": 1, "parent_table_id": "table-1", "image": presentationTestImage(t, color.RGBA{G: 255, A: 255})},
	}
	chunks := slideChunksOf(t, map[string]any{
		"name": "deck.pptx", "output_format": "json", "json": parser.MapsToItems(maps),
	})
	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1: %#v", len(chunks), chunks)
	}
	if got := fmt.Sprint(chunks[0]["slide_number"]); got != "1" {
		t.Fatalf("slide_number lost through typed DocItem input: %#v", chunks[0])
	}
}

func TestPageChunkerCompositeClampsGeometryAndPreservesZOrder(t *testing.T) {
	red := presentationTestImage(t, color.RGBA{R: 255, A: 255})
	blue := presentationTestImage(t, color.RGBA{B: 255, A: 255})
	geometry := map[string]any{
		"slide_canvas_width_emu":   1000,
		"slide_canvas_height_emu":  1000,
		"slide_x_emu":              -100,
		"slide_y_emu":              -100,
		"slide_element_width_emu":  1200,
		"slide_element_height_emu": 1200,
	}
	first := map[string]any{"doc_type_kwd": "image", "slide_number": 1, "image": red}
	second := map[string]any{"doc_type_kwd": "image", "slide_number": 1, "image": blue}
	for key, value := range geometry {
		first[key], second[key] = value, value
	}
	chunks := slideChunksOf(t, map[string]any{"name": "deck.pptx", "output_format": "json", "json": []map[string]any{first, second}})
	if len(chunks) != 1 {
		t.Fatalf("chunks = %#v", chunks)
	}
	preview := decodePresentationTestPreview(t, chunks[0]["image"])
	r, g, b, _ := preview.At(preview.Bounds().Dx()/2, preview.Bounds().Dy()/2).RGBA()
	if b <= r || b <= g {
		t.Fatalf("later blue image did not remain on top: r=%d g=%d b=%d", r, g, b)
	}
}

func TestPageChunkerInvalidMediaDegradesWithoutLosingSearchableText(t *testing.T) {
	chunks := slideChunksOf(t, map[string]any{"name": "deck.pptx", "output_format": "json", "json": []map[string]any{
		{"text": "searchable instructions", "doc_type_kwd": "text", "slide_number": 1},
		{"doc_type_kwd": "image", "slide_number": 1, "image": base64.StdEncoding.EncodeToString([]byte("not an image"))},
	}})
	if len(chunks) != 1 || chunks[0]["text"] != "searchable instructions" {
		t.Fatalf("invalid media lost the slide text: %#v", chunks)
	}
	_ = decodePresentationTestPreview(t, chunks[0]["image"])
}
