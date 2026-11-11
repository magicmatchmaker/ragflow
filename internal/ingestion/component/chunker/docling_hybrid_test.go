// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

package chunker

import (
	"context"
	"strings"
	"sync"
	"testing"

	"ragflow/internal/agent/runtime"
)

func TestDoclingHybridChunkerPassesDoclingAndNativeChunksWithoutResplitting(t *testing.T) {
	component, err := NewDoclingHybridChunker(nil)
	if err != nil {
		t.Fatal(err)
	}
	input := map[string]any{
		"name":          "mixed.csv",
		"output_format": "json",
		"json": []map[string]any{
			{"text": "native CSV row", "doc_type_kwd": "text", "sheet": "Sheet1", "sheet_index": 1},
			{"text": "Docling visual description", "doc_type_kwd": "image", "image": "data:image/png;base64,aW1n", "docling_doc_items": []string{"#/pictures/0"}},
		},
	}
	out, err := component.Invoke(t.Context(), nil, input)
	if err != nil {
		t.Fatal(err)
	}
	chunks, ok := out["chunks"].([]map[string]any)
	if !ok || len(chunks) != 2 {
		t.Fatalf("chunks = %#v, want exactly two unchanged boundaries", out["chunks"])
	}
	if chunks[0]["text"] != "native CSV row" || chunks[1]["text"] != "Docling visual description" {
		t.Fatalf("chunk texts = %#v", chunks)
	}
	if chunks[1]["image"] != "data:image/png;base64,aW1n" {
		t.Fatalf("visual image = %#v", chunks[1]["image"])
	}
}

func TestDoclingHybridChunkerSplitsAndUploadsEveryPictureWithoutIndexingBase64Array(t *testing.T) {
	component, _ := NewDoclingHybridChunker(nil)
	input := map[string]any{
		"name":          "multi.pptx",
		"kb_id":         "kb-1",
		"doc_id":        "doc-1",
		"output_format": "json",
		"json": []map[string]any{{
			"text":                "공식 하이브리드 청크 본문",
			"doc_type_kwd":        "image",
			"docling_chunk_index": 7,
			"docling_headings":    []string{"BCR 점검"},
			"docling_picture_media": []map[string]any{
				{"ref": "#/pictures/0", "image": "data:image/png;base64,aW1hZ2UtMQ==", "positions": [][]any{{1, 10, 20, 30, 40}}, "descriptions": []string{"첫 번째 알람 화면"}},
				{"ref": "#/pictures/1", "image": "data:image/png;base64,aW1hZ2UtMg==", "positions": [][]any{{2, 50, 60, 70, 80}}, "descriptions": []string{"두 번째 조명 화면"}},
			},
		}},
	}

	previous := ChunkImageUploader
	var mu sync.Mutex
	var uploads [][]byte
	ChunkImageUploader = func(_ context.Context, _, _ string, data []byte) (string, error) {
		mu.Lock()
		defer mu.Unlock()
		uploads = append(uploads, append([]byte(nil), data...))
		return "stored-image", nil
	}
	t.Cleanup(func() { ChunkImageUploader = previous })

	out, err := (&imageUploadDecorator{inner: component}).Invoke(t.Context(), nil, input)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	chunks, ok := out["chunks"].([]map[string]any)
	if !ok || len(chunks) != 3 {
		t.Fatalf("chunks = %#v, want one text parent plus one independently uploaded chunk per picture", out["chunks"])
	}
	if len(uploads) != 2 || string(uploads[0]) != "image-1" || string(uploads[1]) != "image-2" {
		t.Fatalf("uploads = %#v, want both picture payloads", uploads)
	}
	for index, chunk := range chunks {
		if _, exists := chunk["image"]; exists {
			t.Fatalf("chunk %d retained raw image bytes: %#v", index, chunk)
		}
		if _, exists := chunk["docling_picture_media"]; exists {
			t.Fatalf("chunk %d leaked internal base64 media array: %#v", index, chunk)
		}
		if index == 0 {
			if chunk["img_id"] != "" || chunk["doc_type_kwd"] != "text" {
				t.Fatalf("parent chunk must be text-only: %#v", chunk)
			}
			continue
		}
		if chunk["img_id"] != "stored-image" || chunk["docling_parent_chunk_index"] != float64(7) {
			t.Fatalf("chunk %d lost uploaded media/source linkage: %#v", index, chunk)
		}
	}
	if chunks[0]["text"] != "공식 하이브리드 청크 본문" || chunks[1]["text"] != "첫 번째 알람 화면" || chunks[2]["text"] != "두 번째 조명 화면" {
		t.Fatalf("text duplication policy = %#v", chunks)
	}
	if strings.Contains(chunks[1]["text"].(string), "두 번째 조명 화면") {
		t.Fatalf("image 1 chunk mixed image 2 description: %#v", chunks[1])
	}
}

func TestDoclingHybridChunkerSameDescriptionWithoutPositionsGetsDistinctMediaIdentity(t *testing.T) {
	component, _ := NewDoclingHybridChunker(nil)
	input := map[string]any{
		"name":          "duplicate-captions.pptx",
		"kb_id":         "kb-1",
		"doc_id":        "doc-identity",
		"output_format": "json",
		"json": []map[string]any{{
			"text":         "두 그림이 포함된 슬라이드",
			"doc_type_kwd": "image",
			"docling_picture_media": []map[string]any{
				{"ref": "#/pictures/0", "image": "data:image/png;base64,c2FtZS1ieXRlcw==", "descriptions": []string{"동일한 설명"}},
				{"ref": "#/pictures/1", "image": "data:image/png;base64,c2FtZS1ieXRlcw==", "descriptions": []string{"동일한 설명"}},
			},
		}},
	}

	previous := ChunkImageUploader
	ChunkImageUploader = func(_ context.Context, _, chunkID string, _ []byte) (string, error) {
		return "stored-" + chunkID, nil
	}
	t.Cleanup(func() { ChunkImageUploader = previous })

	out, err := (&imageUploadDecorator{inner: component}).Invoke(t.Context(), nil, input)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	chunks, ok := out["chunks"].([]map[string]any)
	if !ok || len(chunks) != 3 {
		t.Fatalf("chunks = %#v, want parent plus two picture chunks", out["chunks"])
	}
	firstID, _ := chunks[1]["id"].(string)
	secondID, _ := chunks[2]["id"].(string)
	firstImageID, _ := chunks[1]["img_id"].(string)
	secondImageID, _ := chunks[2]["img_id"].(string)
	if firstID == "" || secondID == "" || firstID == secondID {
		t.Fatalf("picture chunk ids collided: %q %q", firstID, secondID)
	}
	if firstImageID == "" || secondImageID == "" || firstImageID == secondImageID {
		t.Fatalf("picture img_ids collided: %q %q", firstImageID, secondImageID)
	}
	if chunks[1]["text"] != "동일한 설명" || chunks[2]["text"] != "동일한 설명" {
		t.Fatalf("test precondition changed: %#v %#v", chunks[1]["text"], chunks[2]["text"])
	}
}

func TestDoclingHybridChunkerRefusesSilentZeroChunkSuccess(t *testing.T) {
	component, _ := NewDoclingHybridChunker(nil)
	out, err := component.Invoke(t.Context(), nil, map[string]any{
		"name":          "empty.csv",
		"output_format": "json",
		"json":          []map[string]any{},
	})
	if err != nil {
		t.Fatal(err)
	}
	if out["_ERROR"] == nil {
		t.Fatalf("output = %#v, want explicit zero-chunk error", out)
	}
}

func TestDoclingHybridChunkerRegistered(t *testing.T) {
	_, category, _, ok := runtime.DefaultRegistry.Lookup(ComponentNameDoclingHybridChunker)
	if !ok || category != runtime.CategoryIngestion {
		t.Fatalf("registry lookup ok=%v category=%q", ok, category)
	}
}
