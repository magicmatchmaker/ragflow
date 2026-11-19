// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

package chunker

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"ragflow/internal/agent/runtime"
	"ragflow/internal/ingestion/component/schema"
)

const onePixelPNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

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

func TestDoclingHybridChunkerKeepsSemanticChunksAndAddsFullSlidePreview(t *testing.T) {
	component, _ := NewDoclingHybridChunker(nil)
	input := map[string]any{
		"name":          "deck.pptx",
		"doc_id":        "doc-1",
		"output_format": "json",
		"json": []map[string]any{
			{"text": "Docling 의미 청크", "doc_type_kwd": "text", "page_number": 1, "docling_chunk_index": 0},
			{
				"text": "설비 점검 절차", "doc_type_kwd": "text", "slide_number": 1,
				doclingPresentationPreviewSourceKey: true,
				presentationCanvasWidthKey:          1000,
				presentationCanvasHeightKey:         600,
				presentationElementXKey:             20,
				presentationElementYKey:             20,
				presentationElementWidthKey:         800,
				presentationElementHeightKey:        80,
			},
			{
				"text": "현장 설비 사진", "doc_type_kwd": "image", "slide_number": 1,
				"image":                             "data:image/png;base64," + onePixelPNGBase64,
				doclingPresentationPreviewSourceKey: true,
				presentationCanvasWidthKey:          1000,
				presentationCanvasHeightKey:         600,
				presentationElementXKey:             100,
				presentationElementYKey:             120,
				presentationElementWidthKey:         800,
				presentationElementHeightKey:        400,
			},
		},
	}

	out, err := component.Invoke(t.Context(), nil, input)
	if err != nil {
		t.Fatal(err)
	}
	chunks, ok := out["chunks"].([]map[string]any)
	if !ok || len(chunks) != 2 {
		t.Fatalf("chunks = %#v, want one semantic chunk plus one full-slide preview", out["chunks"])
	}
	if chunks[0]["text"] != "Docling 의미 청크" {
		t.Fatalf("semantic boundary changed: %#v", chunks[0])
	}
	preview := chunks[1]
	if preview[doclingSlidePreviewKey] != true || preview[doclingImageRoleKey] != doclingImageRoleFullSlide {
		t.Fatalf("preview metadata = %#v", preview)
	}
	encoded, _ := preview["image"].(string)
	if !strings.HasPrefix(encoded, "data:image/png;base64,") {
		t.Fatalf("full-slide image missing: %#v", preview)
	}
	if _, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(encoded, "data:image/png;base64,")); err != nil {
		t.Fatalf("preview PNG base64: %v", err)
	}
	if _, leaked := preview[doclingPresentationPreviewSourceKey]; leaked {
		t.Fatalf("private preview marker leaked: %#v", preview)
	}
	positions, ok := preview["positions"].([][]float64)
	if !ok || len(positions) != 1 || len(positions[0]) != 5 || positions[0][0] != 1 {
		t.Fatalf("full-slide citation position = %#v, want one page tuple", preview["positions"])
	}
}

func TestDoclingHybridChunkerRecoversPreviewOnlyPresentationAsSemanticAndPreviewChunks(t *testing.T) {
	component, _ := NewDoclingHybridChunker(nil)
	input := map[string]any{
		"name":          "image-only-deck.pptx",
		"kb_id":         "kb-1",
		"doc_id":        "doc-fallback",
		"output_format": "json",
		"json": []map[string]any{
			{
				"text": "OCR recovered alarm reset procedure", "doc_type_kwd": "text", "slide_number": 2,
				doclingPresentationPreviewSourceKey: true,
				presentationCanvasWidthKey:          1000,
				presentationCanvasHeightKey:         600,
				presentationElementXKey:             20,
				presentationElementYKey:             20,
				presentationElementWidthKey:         800,
				presentationElementHeightKey:        80,
			},
			{
				"text": "", "doc_type_kwd": "image", "slide_number": 2,
				"image":                             "data:image/png;base64," + onePixelPNGBase64,
				doclingPresentationPreviewSourceKey: true,
				presentationCanvasWidthKey:          1000,
				presentationCanvasHeightKey:         600,
				presentationElementXKey:             100,
				presentationElementYKey:             120,
				presentationElementWidthKey:         800,
				presentationElementHeightKey:        400,
			},
		},
	}

	out, err := component.Invoke(t.Context(), nil, input)
	if err != nil {
		t.Fatal(err)
	}
	chunks, ok := out["chunks"].([]map[string]any)
	if !ok || len(chunks) != 2 {
		t.Fatalf("output = %#v, want one semantic fallback plus one preview", out)
	}
	semantic, preview := chunks[0], chunks[1]
	if semantic["text"] != "OCR recovered alarm reset procedure" || semantic["doc_type_kwd"] != "text" {
		t.Fatalf("semantic fallback = %#v", semantic)
	}
	if semantic["docling_native_semantic_fallback"] != true || semantic["docling_slide_preview"] != nil {
		t.Fatalf("semantic fallback metadata = %#v", semantic)
	}
	if semantic["img_id"] == "" || preview[doclingSlidePreviewKey] != true {
		t.Fatalf("semantic/preview linkage missing: semantic=%#v preview=%#v", semantic, preview)
	}
}

func TestDoclingPresentationSemanticFallbackKeepsImageOnlySlideSearchable(t *testing.T) {
	page := 4
	previews := []schema.ChunkDoc{{
		Text:       "[Page 4]",
		DocType:    "image",
		PageNumber: &page,
		Image:      "data:image/png;base64," + onePixelPNGBase64,
		Extra:      map[string]json.RawMessage{},
	}}
	setChunkExtraJSON(&previews[0], doclingSlidePreviewKey, true)
	setChunkExtraJSON(&previews[0], doclingImageRoleKey, doclingImageRoleFullSlide)

	semantic := doclingPresentationSemanticFallback(previews, "manual.pptx")
	if len(semantic) != 1 || semantic[0].Text != "manual.pptx — slide 4 visual content" {
		t.Fatalf("semantic = %#v", semantic)
	}
	if semantic[0].DocType != "text" || semantic[0].Image != "" || semantic[0].MediaRef != nil {
		t.Fatalf("semantic retained preview media: %#v", semantic[0])
	}
	if _, exists := semantic[0].Extra[doclingSlidePreviewKey]; exists {
		t.Fatalf("preview marker leaked: %#v", semantic[0].Extra)
	}
}

func TestDoclingPreviewDisplayPolicySuppressesCoverAndTOCOnly(t *testing.T) {
	page1, page2 := 1, 2
	if visible, reason := doclingPreviewDisplayPolicy(schema.ChunkDoc{Text: "회사명과 긴 제품 설명이 포함된 표지 슬라이드", PageNumber: &page1}); visible || reason != doclingPreviewReasonCover {
		t.Fatalf("cover policy = %v %q", visible, reason)
	}
	if visible, reason := doclingPreviewDisplayPolicy(schema.ChunkDoc{Text: "목차 1. 개요 2. 점검", PageNumber: &page2}); visible || reason != doclingPreviewReasonTOC {
		t.Fatalf("TOC policy = %v %q", visible, reason)
	}
	if visible, reason := doclingPreviewDisplayPolicy(schema.ChunkDoc{Text: "알람 발생 시 전원과 인터록 상태를 순서대로 확인한다.", PageNumber: &page2}); !visible || reason != doclingPreviewReasonContent {
		t.Fatalf("content policy = %v %q", visible, reason)
	}
}

func TestLinkDoclingSemanticSlidePreviewsUsesBestContentSlide(t *testing.T) {
	page1, page2 := 1, 2
	chunks := []schema.ChunkDoc{{
		Text:       "Alarm Spec 설정에서 Single과 Label Attachment 사용 범위를 지정한다.",
		DocType:    "text",
		PageNumber: &page1,
		Extra: map[string]json.RawMessage{
			"docling_page_numbers": json.RawMessage(`[1,2]`),
		},
	}}
	previews := []schema.ChunkDoc{
		{Text: "회사 매뉴얼 표지", ID: "cover-id", PageNumber: &page1, Extra: map[string]json.RawMessage{}},
		{Text: "Alarm Spec 설정 Single Label Attachment", ID: "content-id", PageNumber: &page2, Extra: map[string]json.RawMessage{}},
	}
	setChunkExtraJSON(&previews[0], doclingPreviewAutoDisplayKey, false)
	setChunkExtraJSON(&previews[1], doclingPreviewAutoDisplayKey, true)

	linkDoclingSemanticSlidePreviews(chunks, previews, "kb-1")

	if chunks[0].ImgID != "kb-1-content-id" {
		t.Fatalf("semantic chunk img_id = %q, want linked content slide", chunks[0].ImgID)
	}
	var linkedPage int
	if err := json.Unmarshal(chunks[0].Extra["docling_linked_slide_page"], &linkedPage); err != nil || linkedPage != 2 {
		t.Fatalf("linked slide page = %d, err=%v", linkedPage, err)
	}
}

func TestDoclingHybridChunkerRegistered(t *testing.T) {
	_, category, _, ok := runtime.DefaultRegistry.Lookup(ComponentNameDoclingHybridChunker)
	if !ok || category != runtime.CategoryIngestion {
		t.Fatalf("registry lookup ok=%v category=%q", ok, category)
	}
}
