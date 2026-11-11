// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

package component

import (
	"strings"
	"testing"
	"time"

	"ragflow/internal/ingestion/component/schema"
	"ragflow/internal/ingestion/task/indexdoc"
)

func TestDoclingVisualDescriptionSurvivesEmbeddingAndGenerationIndexContent(t *testing.T) {
	const generationText = "설비 알람 화면. 빨간 표시등과 ALARM 37이 보인다."
	chunks := []schema.ChunkDoc{{
		Text:    generationText,
		DocType: "image",
		Summary: "롤프레스 BCR 조명 불량 진단 문맥\n\n" + generationText,
	}}
	if embedded := concatFields(chunks[0], []string{"summary"}); !strings.Contains(embedded, "롤프레스") || !strings.Contains(embedded, "ALARM 37") {
		t.Fatalf("embedding input = %q, want contextual prefix and visual text", embedded)
	}

	rows := schema.ChunkDocsToMaps(chunks)
	if _, err := indexdoc.ProcessChunksForPipeline(rows, "doc-1", "alarm.jpg", time.Unix(1, 0)); err != nil {
		t.Fatalf("ProcessChunksForPipeline: %v", err)
	}
	if got := rows[0]["content_with_weight"]; got != generationText {
		t.Fatalf("content_with_weight = %#v; answer model would lose Docling visual description", got)
	}
	if _, exists := rows[0]["summary"]; exists {
		t.Fatalf("transient contextual search text leaked into generation content: %#v", rows[0])
	}
}
