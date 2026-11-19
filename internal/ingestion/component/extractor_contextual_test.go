// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

package component

import (
	"context"
	"errors"
	"strings"
	"testing"

	"ragflow/internal/common"
)

func TestExtractorContextualRetrievalKeepsGenerationTextAndSeparatesSearchText(t *testing.T) {
	stub := withStubChatInvoker(t,
		stubResponse{Content: "롤프레스 BCR 조명 불량 진단 문맥"},
		stubResponse{Content: "롤프레스 BCR 조명 불량 진단 문맥"},
	)
	component, err := NewExtractorComponent(map[string]any{
		"llm_id":     "qwen3.8-flash-next@openai",
		"contextual": map[string]any{"enabled": true},
		"summary":    map[string]any{"enabled": false},
	})
	if err != nil {
		t.Fatal(err)
	}
	chunks := []map[string]any{
		{"id": "c1", "text": "빨간 표시등과 ALARM 37이 보인다.", "docling_raw_text": "ALARM 37"},
		{"id": "c2", "text": "조명 컨트롤러 밝기를 높인다."},
	}
	out, err := component.Invoke(t.Context(), nil, map[string]any{"chunks": chunks})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	got := out["chunks"].([]map[string]any)
	if got[0]["text"] != "빨간 표시등과 ALARM 37이 보인다." {
		t.Fatalf("generation text changed: %#v", got[0]["text"])
	}
	if got[0]["docling_raw_text"] != "ALARM 37" {
		t.Fatalf("raw provenance changed: %#v", got[0]["docling_raw_text"])
	}
	wantContextual := "롤프레스 BCR 조명 불량 진단 문맥\n\n빨간 표시등과 ALARM 37이 보인다."
	if got[0][common.ContextualContentField] != wantContextual {
		t.Fatalf("contextual_with_weight = %#v, want contextual prefix + generation text", got[0][common.ContextualContentField])
	}
	if _, exists := got[0]["summary"]; exists {
		t.Fatalf("contextual retrieval must not reuse the summary channel: %#v", got[0])
	}
	if calls := stub.Calls(); calls != 2 {
		t.Fatalf("LLM calls = %d, want one per chunk", calls)
	}
	stub.mu.Lock()
	defer stub.mu.Unlock()
	var sharedDocumentPrefix string
	for _, request := range stub.requests {
		if request.Temperature == nil || *request.Temperature != 0 {
			t.Fatalf("temperature = %#v, want deterministic zero", request.Temperature)
		}
		user := request.Messages[len(request.Messages)-1].Content
		for _, want := range []string{"<chunk>", "<document>", "ALARM 37", "조명 컨트롤러"} {
			if !strings.Contains(user, want) {
				t.Fatalf("contextual prompt missing %q: %s", want, user)
			}
		}
		documentAt := strings.Index(user, "<document>")
		chunkAt := strings.LastIndex(user, "<chunk>")
		if documentAt < 0 || chunkAt <= documentAt || !strings.HasSuffix(user, "</chunk>") {
			t.Fatalf("contextual prompt must put common document first and target chunk last: %s", user)
		}
		prefix := user[:chunkAt]
		if sharedDocumentPrefix == "" {
			sharedDocumentPrefix = prefix
		} else if prefix != sharedDocumentPrefix {
			t.Fatalf("document prefix differs across chunk calls; vLLM prefix cache cannot be reused")
		}
	}
}

func TestExtractorContextualRetrievalSkipsDerivedSlidePreview(t *testing.T) {
	stub := withStubChatInvoker(t, stubResponse{Content: "문서 의미 문맥"})
	component, err := NewExtractorComponent(map[string]any{
		"llm_id":     "qwen3.8-flash-next@openai",
		"contextual": map[string]any{"enabled": true},
	})
	if err != nil {
		t.Fatal(err)
	}
	chunks := []map[string]any{
		{"id": "semantic-1", "text": "ALARM 37의 조치 순서"},
		{"id": "slide-1", "text": "ALARM 37의 조치 순서", doclingSlidePreviewChunkKey: true, "img_id": "slide-image"},
	}
	out, err := component.Invoke(t.Context(), nil, map[string]any{"chunks": chunks})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	got := out["chunks"].([]map[string]any)
	if calls := stub.Calls(); calls != 1 {
		t.Fatalf("LLM calls = %d, want semantic chunks only", calls)
	}
	if got[0][common.ContextualContentField] != "문서 의미 문맥\n\nALARM 37의 조치 순서" {
		t.Fatalf("semantic context = %#v", got[0][common.ContextualContentField])
	}
	if got[1][common.ContextualContentField] != "ALARM 37의 조치 순서" {
		t.Fatalf("preview search text = %#v", got[1][common.ContextualContentField])
	}
	stub.mu.Lock()
	defer stub.mu.Unlock()
	prompt := stub.requests[0].Messages[len(stub.requests[0].Messages)-1].Content
	if strings.Count(prompt, "ALARM 37의 조치 순서") != 2 {
		t.Fatalf("prompt duplicated preview text: %s", prompt)
	}
}

func TestExtractorContextualRetrievalRejectsSilentContextTruncation(t *testing.T) {
	SetExtractorContextLengthOverride(func(_ context.Context, _ string) int { return 200 })
	t.Cleanup(func() { SetExtractorContextLengthOverride(nil) })
	stub := withStubChatInvoker(t, stubResponse{Content: "must not be called"})
	c := &ExtractorComponent{}
	document := strings.Repeat("shared document context ", 4000)
	target := "TARGET-ALARM-37-MUST-SURVIVE"
	_, err := c.callText(t.Context(), nil, extractorInputs{
		llmID:                  "test@test",
		rejectPromptTruncation: true,
	}, contextualRetrievalPrompt, contextualRetrievalUserPrompt(target, document))
	if err == nil || !strings.Contains(err.Error(), "refusing silent") {
		t.Fatalf("error = %v, want explicit contextual overflow failure", err)
	}
	if calls := stub.Calls(); calls != 0 {
		t.Fatalf("invoker calls = %d, want prompt rejected before provider call", calls)
	}
}

func TestContextualDocumentWindowBoundsLargeManualAndKeepsOpeningAndLocalSection(t *testing.T) {
	chunks := make([]map[string]any, 24)
	for index := range chunks {
		chunks[index] = map[string]any{
			"id":   "chunk",
			"text": strings.Repeat(string(rune('A'+index)), 180),
		}
	}
	const target = 18
	got := contextualDocumentWindow(chunks, target, 1100)
	if len([]rune(got)) > 1100 {
		t.Fatalf("window runes = %d, want <= 1100", len([]rune(got)))
	}
	if !strings.Contains(got, strings.Repeat("A", 40)) {
		t.Fatal("window omitted document opening")
	}
	if !strings.Contains(got, strings.Repeat(string(rune('A'+target)), 40)) {
		t.Fatal("window omitted target chunk")
	}
	if !strings.Contains(got, contextualDocumentOmissionMarker) {
		t.Fatal("large window must mark omitted context explicitly")
	}
}

func TestExtractorContextualRetrievalFailureIsNotSilentlySkipped(t *testing.T) {
	withStubChatInvoker(t, stubResponse{Err: errors.New("401 unauthorized")})
	component, err := NewExtractorComponent(map[string]any{
		"llm_id":     "qwen3.8-flash-next@openai",
		"contextual": map[string]any{"enabled": true},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = component.Invoke(t.Context(), nil, map[string]any{
		"chunks": []map[string]any{{"id": "c1", "text": "body"}},
	})
	if err == nil || !strings.Contains(err.Error(), "contextual retrieval chunk 0") {
		t.Fatalf("error = %v, want hard contextual failure", err)
	}
}

func TestExtractorAllowsContextualAndSummaryOnSeparateFields(t *testing.T) {
	_, err := NewExtractorComponent(map[string]any{
		"contextual": map[string]any{"enabled": true},
		"summary":    map[string]any{"enabled": true},
	})
	if err != nil {
		t.Fatalf("separate contextual_with_weight and summary fields should validate: %v", err)
	}
}
