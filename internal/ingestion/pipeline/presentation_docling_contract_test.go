// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

package pipeline

import (
	"encoding/json"
	"testing"
)

func TestPresentationTemplateRemainsLegacyPresentationPipeline(t *testing.T) {
	dsl, err := LoadBuiltinDSL("presentation")
	if err != nil {
		t.Fatalf("LoadBuiltinDSL: %v", err)
	}
	var root map[string]any
	if err := json.Unmarshal([]byte(dsl), &root); err != nil {
		t.Fatalf("decode presentation DSL: %v", err)
	}
	components, ok := root["components"].(map[string]any)
	if !ok {
		t.Fatal("presentation DSL has no components")
	}
	if _, exists := components["PageChunker:HappyHillsGlow"]; !exists {
		t.Fatal("presentation DSL no longer routes through PageChunker")
	}
	if _, exists := components["DoclingHybridChunker:HappyHillsGlow"]; exists {
		t.Fatal("presentation DSL was mutated to use DoclingHybridChunker")
	}

	params := componentParamsForContract(t, components, "Parser:HipSignsRhyme")
	for _, family := range []string{"pdf", "slides"} {
		setup, ok := params[family].(map[string]any)
		if !ok {
			t.Fatalf("parser has no %s setup", family)
		}
		if got := setup["parse_method"]; got != "DeepDOC" {
			t.Fatalf("parser %s parse_method = %v, want DeepDOC", family, got)
		}
	}

	tokenizer := componentParamsForContract(t, components, "Tokenizer:TallTreesDance")
	if got := tokenizer["fields"]; got != "text" {
		t.Fatalf("Tokenizer fields = %v, want legacy text channel", got)
	}
	extractor := componentParamsForContract(t, components, "Extractor:AutoExtractDefault")
	if _, exists := extractor["contextual"]; exists {
		t.Fatalf("legacy presentation Extractor unexpectedly enables contextual retrieval: %#v", extractor["contextual"])
	}
}

func TestDoclingContextualTemplateIsCompleteMigrationSource(t *testing.T) {
	dsl, err := LoadBuiltinDSL("docling_contextual")
	if err != nil {
		t.Fatalf("LoadBuiltinDSL: %v", err)
	}
	var root map[string]any
	if err := json.Unmarshal([]byte(dsl), &root); err != nil {
		t.Fatalf("decode docling_contextual DSL: %v", err)
	}
	components, ok := root["components"].(map[string]any)
	if !ok {
		t.Fatal("docling_contextual DSL has no components")
	}
	if _, exists := components["PageChunker:HappyHillsGlow"]; exists {
		t.Fatal("docling_contextual DSL routes through legacy PageChunker")
	}
	if _, exists := components["DoclingHybridChunker:DoclingContextual"]; !exists {
		t.Fatal("docling_contextual DSL has no DoclingHybridChunker")
	}

	params := componentParamsForContract(t, components, "Parser:DoclingContextual")
	for _, family := range []string{"pdf", "slides", "docx", "spreadsheet", "markdown", "image"} {
		setup, ok := params[family].(map[string]any)
		if !ok {
			t.Fatalf("parser has no %s setup", family)
		}
		if got := setup["parse_method"]; got != "DoclingHybrid" {
			t.Fatalf("parser %s parse_method = %v, want DoclingHybrid", family, got)
		}
	}
	spreadsheet := params["spreadsheet"].(map[string]any)
	if !contractStringSliceContains(spreadsheet["suffix"], "csv") {
		t.Fatalf("spreadsheet suffix = %#v, want CSV routed through verified Docling endpoint", spreadsheet["suffix"])
	}

	tokenizer := componentParamsForContract(t, components, "Tokenizer:DoclingContextual")
	if got := tokenizer["fields"]; got != "contextual_with_weight" {
		t.Fatalf("Tokenizer fields = %v, want dedicated contextual content", got)
	}
	extractor := componentParamsForContract(t, components, "Extractor:DoclingContextual")
	contextual, _ := extractor["contextual"].(map[string]any)
	if contextual["enabled"] != true {
		t.Fatalf("Extractor contextual = %#v, want enabled", extractor["contextual"])
	}
	summary, _ := extractor["summary"].(map[string]any)
	if summary["enabled"] != false {
		t.Fatalf("Extractor summary = %#v, want disabled to avoid channel collision", extractor["summary"])
	}
	graph, _ := root["graph"].(map[string]any)
	nodes, _ := graph["nodes"].([]any)
	wantNodes := map[string]bool{
		"File":                                   false,
		"Parser:DoclingContextual":               false,
		"DoclingHybridChunker:DoclingContextual": false,
		"Extractor:DoclingContextual":            false,
		"Tokenizer:DoclingContextual":            false,
	}
	for _, raw := range nodes {
		node, _ := raw.(map[string]any)
		id, _ := node["id"].(string)
		if _, exists := wantNodes[id]; exists {
			wantNodes[id] = true
		}
		data, _ := node["data"].(map[string]any)
		form, _ := data["form"].(map[string]any)
		switch id {
		case "Extractor:DoclingContextual":
			contextualForm, _ := form["contextual"].(map[string]any)
			if contextualForm["enabled"] != true {
				t.Errorf("Extractor graph form does not expose contextual=true: %#v", form)
			}
		case "Tokenizer:DoclingContextual":
			if form["fields"] != "contextual_with_weight" {
				t.Errorf("Tokenizer graph form fields = %#v, want contextual_with_weight", form["fields"])
			}
		}
	}
	for id, found := range wantNodes {
		if !found {
			t.Errorf("docling_contextual graph missing node %s", id)
		}
	}
	edges, _ := graph["edges"].([]any)
	if len(edges) != 4 {
		t.Fatalf("docling_contextual graph edges = %d, want full 4-stage chain", len(edges))
	}
}

func componentParamsForContract(t *testing.T, components map[string]any, id string) map[string]any {
	t.Helper()
	component, ok := components[id].(map[string]any)
	if !ok {
		t.Fatalf("DSL has no %s", id)
	}
	obj, ok := component["obj"].(map[string]any)
	if !ok {
		t.Fatalf("component %s has no obj", id)
	}
	params, ok := obj["params"].(map[string]any)
	if !ok {
		t.Fatalf("component %s has no params", id)
	}
	return params
}

func contractStringSliceContains(value any, want string) bool {
	values, _ := value.([]any)
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
