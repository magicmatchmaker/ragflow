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
	"errors"
	"image/color"
	"strings"
	"testing"

	deepdocpdf "ragflow/internal/deepdoc/parser/pdf"
	deepdoctype "ragflow/internal/deepdoc/parser/type"
	component "ragflow/internal/ingestion/component"
	"ragflow/internal/ingestion/component/schema"
)

type presentationRecordingEmbedder struct {
	inputs [][]string
}

func (*presentationRecordingEmbedder) MaxTokens() int { return 8192 }

func (*presentationRecordingEmbedder) BatchSize() int { return 8 }

func (e *presentationRecordingEmbedder) Encode(_ context.Context, texts []string) ([]component.EmbeddingResult, error) {
	e.inputs = append(e.inputs, append([]string(nil), texts...))
	results := make([]component.EmbeddingResult, len(texts))
	for i := range results {
		results[i] = component.EmbeddingResult{Vector: []float64{1, 0, 0, 0}, TokenCount: len(texts[i])}
	}
	return results, nil
}

func usePresentationOCRAnalyzer(t *testing.T, analyzer deepdoctype.DocAnalyzer, available bool) {
	t.Helper()
	previous := deepdoctype.NativeDocAnalyzerFactory
	t.Cleanup(func() { deepdoctype.NativeDocAnalyzerFactory = previous })
	deepdoctype.SetNativeDocAnalyzerFactory(func() (deepdoctype.DocAnalyzer, bool) {
		return analyzer, available
	})
}

func presentationOCRAnalyzer(text string) *deepdocpdf.MockDocAnalyzer {
	return &deepdocpdf.MockDocAnalyzer{
		Healthy: true,
		OCRBoxes: []deepdoctype.OCRBox{{
			X0: 1, Y0: 1,
			X1: 40, Y1: 1,
			X2: 40, Y2: 20,
			X3: 1, Y3: 20,
		}},
		OCRTexts: []deepdoctype.OCRText{{Text: text, Confidence: 0.99}},
	}
}

func presentationImageOnlyInputs(t *testing.T, text string) map[string]any {
	t.Helper()
	item := map[string]any{
		"doc_type_kwd": "image",
		"slide_number": 1,
		"image":        presentationTestImage(t, color.RGBA{R: 28, G: 64, B: 92, A: 255}),
	}
	if text != "" {
		item["text"] = text
	}
	return map[string]any{
		"name":          "synthetic.pptx",
		"file_type":     "pptx",
		"output_format": "json",
		"json":          []map[string]any{item},
	}
}

func TestPageChunkerImageOnlySlideOCRReachesEmbeddingInput(t *testing.T) {
	const recognized = "synthetic safety instruction"
	usePresentationOCRAnalyzer(t, presentationOCRAnalyzer(recognized), true)

	page, err := NewPageChunker(nil)
	if err != nil {
		t.Fatal(err)
	}
	pageOut, err := page.Invoke(t.Context(), nil, presentationImageOnlyInputs(t, ""))
	if err != nil {
		t.Fatalf("PageChunker.Invoke: %v", err)
	}
	pageChunks, ok := schema.ChunkCarrierToMaps(pageOut["chunks"])
	if !ok || len(pageChunks) != 1 {
		t.Fatalf("page chunks = %#v", pageOut["chunks"])
	}
	if got, _ := pageChunks[0]["text"].(string); !strings.Contains(got, recognized) {
		t.Fatalf("OCR text did not reach the slide chunk: %q", got)
	}
	decodePresentationTestPreview(t, pageChunks[0]["image"])

	embedder := &presentationRecordingEmbedder{}
	tokenizer, err := component.NewTokenizerComponentWithResolver(
		map[string]any{
			"search_method":        []any{"embedding"},
			"filename_embd_weight": 0,
		},
		func(context.Context, string, string) (component.Embedder, string, error) {
			return embedder, "presentation-test", nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tokenizer.Invoke(t.Context(), nil, map[string]any{
		"name":          "synthetic.pptx",
		"kb_id":         "kb-presentation-test",
		"output_format": "chunks",
		"chunks":        pageChunks,
	}); err != nil {
		t.Fatalf("Tokenizer.Invoke: %v", err)
	}
	found := false
	for _, batch := range embedder.inputs {
		for _, input := range batch {
			if strings.Contains(input, recognized) {
				found = true
			}
		}
	}
	if !found {
		t.Fatalf("OCR text did not reach Embedder.Encode: %#v", embedder.inputs)
	}
}

func TestPageChunkerPresentationOCRDeduplicatesExistingText(t *testing.T) {
	usePresentationOCRAnalyzer(t, presentationOCRAnalyzer("Existing   Label"), true)
	chunks := slideChunksOf(t, presentationImageOnlyInputs(t, "Existing Label"))
	if len(chunks) != 1 {
		t.Fatalf("chunks = %#v", chunks)
	}
	if got := chunks[0]["text"]; got != "Existing Label" {
		t.Fatalf("deduplicated text = %q, want the parser text once", got)
	}
	decodePresentationTestPreview(t, chunks[0]["image"])
}

func TestPageChunkerPresentationOCRUnavailableKeepsTextAndPreview(t *testing.T) {
	usePresentationOCRAnalyzer(t, nil, false)
	chunks := slideChunksOf(t, presentationImageOnlyInputs(t, "parser text remains searchable"))
	if len(chunks) != 1 || chunks[0]["text"] != "parser text remains searchable" {
		t.Fatalf("unavailable analyzer lost parser text: %#v", chunks)
	}
	decodePresentationTestPreview(t, chunks[0]["image"])
}

func TestPageChunkerPresentationOCRFactoryPanicKeepsTextAndPreview(t *testing.T) {
	previous := deepdoctype.NativeDocAnalyzerFactory
	t.Cleanup(func() { deepdoctype.NativeDocAnalyzerFactory = previous })
	deepdoctype.SetNativeDocAnalyzerFactory(func() (deepdoctype.DocAnalyzer, bool) {
		panic("synthetic analyzer initialization panic")
	})
	chunks := slideChunksOf(t, presentationImageOnlyInputs(t, "parser text survives factory panic"))
	if len(chunks) != 1 || chunks[0]["text"] != "parser text survives factory panic" {
		t.Fatalf("factory panic lost parser text: %#v", chunks)
	}
	decodePresentationTestPreview(t, chunks[0]["image"])
}

func TestPageChunkerPresentationOCRFailureKeepsTextAndPreview(t *testing.T) {
	analyzer := presentationOCRAnalyzer("unused")
	analyzer.OCRDetectErr = errors.New("synthetic OCR failure")
	usePresentationOCRAnalyzer(t, analyzer, true)
	chunks := slideChunksOf(t, presentationImageOnlyInputs(t, "parser fallback"))
	if len(chunks) != 1 || chunks[0]["text"] != "parser fallback" {
		t.Fatalf("failed OCR lost parser text: %#v", chunks)
	}
	decodePresentationTestPreview(t, chunks[0]["image"])
}

func TestMergePresentationOCRCapsAddedText(t *testing.T) {
	got := mergePresentationOCR("base", strings.Repeat("x", maxPresentationOCRRunes+200))
	if added := len([]rune(strings.TrimPrefix(got, "base\n\n"))); added != maxPresentationOCRRunes {
		t.Fatalf("OCR contribution runes = %d, want %d", added, maxPresentationOCRRunes)
	}
}
