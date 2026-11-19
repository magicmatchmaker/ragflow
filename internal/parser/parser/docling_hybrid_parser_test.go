// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

package parser

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"ragflow/internal/common"
	"ragflow/internal/utility"
)

type doclingRoundTripFunc func(*http.Request) (*http.Response, error)

func (fn doclingRoundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

func resetDoclingClientCache(t *testing.T) {
	t.Helper()
	doclingHTTPClientCache.Lock()
	if doclingHTTPClientCache.client != nil {
		doclingHTTPClientCache.client.CloseIdleConnections()
	}
	doclingHTTPClientCache.client = nil
	doclingHTTPClientCache.Unlock()
	t.Cleanup(func() {
		doclingHTTPClientCache.Lock()
		if doclingHTTPClientCache.client != nil {
			doclingHTTPClientCache.client.CloseIdleConnections()
		}
		doclingHTTPClientCache.client = nil
		doclingHTTPClientCache.Unlock()
	})
}

func TestDoclingHybridParserLiveWireContentAndPictureProvenance(t *testing.T) {
	resetDoclingClientCache(t)
	secretPath := filepath.Join(t.TempDir(), "dgx_api_key")
	if err := os.WriteFile(secretPath, []byte("dgx-secret\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("DOCLING_PICTURE_DESCRIPTION_API_KEY_FILE", secretPath)
	t.Setenv("DOCLING_PICTURE_DESCRIPTION_URL", "http://10.90.0.2:8080/v1/chat/completions")
	t.Setenv("DOCLING_PICTURE_DESCRIPTION_MODEL", "qwen3.8-flash-next")
	t.Setenv("DOCLING_HYBRID_TOKENIZER", "Qwen/Qwen3-VL-Embedding-2B")
	t.Setenv("DOCLING_HYBRID_MAX_TOKENS", "800")

	var request map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chunk/hybrid/source" {
			t.Errorf("request = %s %s, want POST /v1/chunk/hybrid/source", r.Method, r.URL.Path)
			http.NotFound(w, r)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
  "chunks":[{
    "filename":"alarm.jpg","chunk_index":0,
    "text":"ALARM 37",
    "raw_text":"ALARM 37","num_tokens":17,
    "headings":["설비 점검"],"captions":["알람 화면"],
    "doc_items":["#/texts/0","#/pictures/0"],"page_numbers":[1]
  }],
  "documents":[{
    "kind":"ExportResult",
    "status":"success",
    "content":{"filename":"alarm.jpg","json_content":{
      "pages":{"1":{"page_no":1,"size":{"width":1000,"height":800}}},
      "texts":[{"self_ref":"#/texts/0","text":"ALARM 37","prov":[{"page_no":1,"bbox":{"l":100,"t":200,"r":400,"b":260,"coord_origin":"TOPLEFT"}}]}],
      "pictures":[{"self_ref":"#/pictures/0","label":"picture","prov":[{"page_no":1,"bbox":{"l":50,"t":700,"r":950,"b":100,"coord_origin":"BOTTOMLEFT"}}],"image":{"uri":"data:image/png;base64,aW1hZ2U="},"meta":{"description":{"text":"빨간 경고등과 ALARM 37 코드가 표시된 HMI 화면","confidence":0.99}}}]
    }}
  }],
  "processing_time":1.25
}`))
	}))
	defer server.Close()
	t.Setenv("DOCLING_SERVER_URL", doclingTrustedOrigin)
	previousLookup := common.LookupHost
	common.LookupHost = func(host string) ([]string, error) {
		if host != "docling-serve" {
			t.Fatalf("resolved host = %q, want docling-serve", host)
		}
		return []string{"127.0.0.1"}, nil
	}
	previousPinned := utility.PinnedHTTPClient
	utility.PinnedHTTPClient = func(hostname, resolvedIP string, timeout time.Duration) *http.Client {
		if hostname != "docling-serve" || resolvedIP != "127.0.0.1" || timeout != 0 {
			t.Fatalf("pinned client args = %q %q %s", hostname, resolvedIP, timeout)
		}
		return &http.Client{Transport: doclingRoundTripFunc(func(req *http.Request) (*http.Response, error) {
			cloned := req.Clone(req.Context())
			target := *req.URL
			target.Scheme = "http"
			target.Host = strings.TrimPrefix(server.URL, "http://")
			cloned.URL = &target
			return server.Client().Transport.RoundTrip(cloned)
		})}
	}
	t.Cleanup(func() {
		common.LookupHost = previousLookup
		utility.PinnedHTTPClient = previousPinned
	})

	p := NewDoclingHybridParser(utility.FileTypeVISUAL)
	p.ConfigureFromSetup(map[string]any{
		"docling_server_url":                  "http://attacker.invalid",
		"docling_hybrid_tokenizer":            "attacker/tokenizer",
		"docling_hybrid_max_tokens":           1,
		"docling_picture_description_url":     "https://attacker.invalid/steal-secret",
		"docling_picture_description_model":   "attacker-model",
		"docling_picture_description_api_key": "plaintext-must-be-ignored",
	})
	result := p.ParseWithResult(t.Context(), "alarm.jpg", []byte("jpeg-bytes"))
	if result.Err != nil {
		t.Fatalf("ParseWithResult: %v", result.Err)
	}
	if len(result.JSON) != 1 {
		t.Fatalf("JSON len = %d, want 1", len(result.JSON))
	}
	item := result.JSON[0].ToMap()
	text, _ := item["text"].(string)
	if !strings.Contains(text, "ALARM 37") || !strings.Contains(text, "빨간 경고등과 ALARM 37 코드가 표시된 HMI 화면") {
		t.Fatalf("text = %#v; generation content must include Hybrid text and meta.description", item["text"])
	}
	if got := item["docling_raw_text"]; got != "ALARM 37" {
		t.Fatalf("docling_raw_text = %#v, want raw source text", got)
	}
	if got := item["image"]; got != "data:image/png;base64,aW1hZ2U=" {
		t.Fatalf("image = %#v", got)
	}
	descriptions, _ := item["docling_picture_descriptions"].([]string)
	if len(descriptions) != 1 || !strings.Contains(descriptions[0], "ALARM 37") {
		t.Fatalf("picture descriptions = %#v", item["docling_picture_descriptions"])
	}
	positions, _ := item["positions"].([]any)
	if len(positions) != 2 {
		t.Fatalf("positions = %#v, want text and picture provenance", item["positions"])
	}

	if got := request["include_converted_doc"]; got != true {
		t.Fatalf("include_converted_doc = %#v", got)
	}
	sources, _ := request["sources"].([]any)
	source, _ := sources[0].(map[string]any)
	if source["kind"] != "file" || source["filename"] != "alarm.jpg" {
		t.Fatalf("source = %#v", source)
	}
	decoded, err := base64.StdEncoding.DecodeString(source["base64_string"].(string))
	if err != nil || string(decoded) != "jpeg-bytes" {
		t.Fatalf("base64 source = %q err=%v", decoded, err)
	}
	convert, _ := request["convert_options"].(map[string]any)
	if convert["image_export_mode"] != "embedded" || convert["do_picture_description"] != true || convert["picture_description_area_threshold"] != float64(0) {
		t.Fatalf("convert_options = %#v", convert)
	}
	pictureAPI, _ := convert["picture_description_api"].(map[string]any)
	if pictureAPI["url"] != "http://10.90.0.2:8080/v1/chat/completions" || pictureAPI["timeout"] != float64(defaultDoclingPictureDescriptionTimeout) {
		t.Fatalf("picture api operator settings = %#v", pictureAPI)
	}
	headers, _ := pictureAPI["headers"].(map[string]any)
	if headers["Authorization"] != "Bearer dgx-secret" {
		t.Fatalf("picture headers = %#v", headers)
	}
	params, _ := pictureAPI["params"].(map[string]any)
	if params["model"] != "qwen3.8-flash-next" || pictureAPI["concurrency"] != float64(2) {
		t.Fatalf("picture api = %#v", pictureAPI)
	}
	chunking, _ := request["chunking_options"].(map[string]any)
	if chunking["chunker"] != "hybrid" || chunking["include_raw_text"] != true || chunking["tokenizer"] != "Qwen/Qwen3-VL-Embedding-2B" {
		t.Fatalf("chunking_options = %#v", chunking)
	}
}

func TestDoclingHybridParserConfiguredSecretFileFailureIsFatal(t *testing.T) {
	t.Setenv("DOCLING_PICTURE_DESCRIPTION_API_KEY_FILE", filepath.Join(t.TempDir(), "missing"))
	t.Setenv("DOCLING_SERVER_URL", "http://docling.invalid")
	t.Setenv("DOCLING_PICTURE_DESCRIPTION_URL", "http://dgx.invalid/v1/chat/completions")
	p := NewDoclingHybridParser(utility.FileTypePPTX)
	p.ConfigureFromSetup(map[string]any{
		"docling_picture_description_api_key": "plaintext-must-not-bypass-missing-file",
	})
	result := p.ParseWithResult(t.Context(), "deck.pptx", []byte("pptx"))
	if result.Err == nil || !strings.Contains(result.Err.Error(), "secret file") {
		t.Fatalf("error = %v, want secret file failure before network call", result.Err)
	}
}

func TestDoclingHybridParserRejectsEmptySource(t *testing.T) {
	p := NewDoclingHybridParser(utility.FileTypePDF)
	result := p.ParseWithResult(t.Context(), "empty.pdf", nil)
	if result.Err == nil || !strings.Contains(result.Err.Error(), "is empty") {
		t.Fatalf("error = %v, want explicit empty-source failure", result.Err)
	}
}

func TestDoclingPictureDescriptionDoesNotSkipSmallTechnicalImagesByDefault(t *testing.T) {
	t.Setenv("DOCLING_PICTURE_DESCRIPTION_URL", "http://10.90.0.2:8080/v1/chat/completions")
	p := NewDoclingHybridParser(utility.FileTypeVISUAL)
	payload, err := p.requestPayload("tiny-alarm-icon.png", []byte("small-image"))
	if err != nil {
		t.Fatal(err)
	}
	convert := payload["convert_options"].(map[string]any)
	if got := convert["picture_description_area_threshold"]; got != float64(0) {
		t.Fatalf("picture_description_area_threshold = %#v, want 0 so small technical images are described", got)
	}
	api := convert["picture_description_api"].(map[string]any)
	if api["timeout"] != defaultDoclingPictureDescriptionTimeout || api["concurrency"] != defaultDoclingPictureDescriptionWorkers {
		t.Fatalf("picture API defaults = %#v", api)
	}
}

func TestDoclingPPTXRequestUsesNativePreviewInsteadOfEmbeddingAllSourceImages(t *testing.T) {
	p := NewDoclingHybridParser(utility.FileTypePPTX)
	payload, err := p.requestPayload("large-deck.pptx", []byte("pptx"))
	if err != nil {
		t.Fatal(err)
	}
	convert := payload["convert_options"].(map[string]any)
	if convert["include_images"] != false || convert["image_export_mode"] != "placeholder" {
		t.Fatalf("PPTX convert options = %#v, want native-preview media ownership", convert)
	}
}

func TestDoclingMarkdownRequestUsesMarkdownInputFormat(t *testing.T) {
	p := NewDoclingHybridParser(utility.FileTypeMarkdown)
	payload, err := p.requestPayload("knowledge.md", []byte("# Knowledge"))
	if err != nil {
		t.Fatal(err)
	}
	convert := payload["convert_options"].(map[string]any)
	formats := convert["from_formats"].([]string)
	if len(formats) != 1 || formats[0] != "md" {
		t.Fatalf("Markdown from_formats = %#v, want [md]", formats)
	}
}

func TestValidateDoclingHybridDocumentRejectsNonSuccessAndErrors(t *testing.T) {
	page := 3
	for _, tc := range []struct {
		name string
		doc  doclingHybridDocumentResult
		want string
	}{
		{name: "failure", doc: doclingHybridDocumentResult{Status: "failure"}, want: `status="failure"`},
		{name: "partial", doc: doclingHybridDocumentResult{Status: "partial_success"}, want: `status="partial_success"`},
		{name: "skipped", doc: doclingHybridDocumentResult{Status: "skipped"}, want: `status="skipped"`},
		{name: "success with errors", doc: doclingHybridDocumentResult{Status: "success", Errors: []doclingHybridError{{ModuleName: "ocr", ErrorMessage: "Bearer secret and base64 source", PageNo: &page}}}, want: "page 3: ocr: conversion error"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validateDoclingHybridDocument(tc.doc)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
	if err := validateDoclingHybridDocument(doclingHybridDocumentResult{Status: "success", Errors: []doclingHybridError{{ErrorMessage: "Bearer secret and base64 source"}}}); err == nil || strings.Contains(err.Error(), "secret") || strings.Contains(err.Error(), "base64") {
		t.Fatalf("conversion error leaked untrusted remote message: %v", err)
	}
	if err := validateDoclingHybridDocument(doclingHybridDocumentResult{Status: "success"}); err != nil {
		t.Fatalf("success rejected: %v", err)
	}
}

func TestDoclingHybridRequestRetriesTransientHTTPFailure(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			http.Error(w, "temporary", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"chunks":[],"documents":[],"processing_time":0}`))
	}))
	defer server.Close()
	t.Setenv("DOCLING_REQUEST_RETRY_MAX", "1")
	previousDelay := doclingRequestRetryDelay
	doclingRequestRetryDelay = time.Millisecond
	t.Cleanup(func() { doclingRequestRetryDelay = previousDelay })
	p := NewDoclingHybridParser(utility.FileTypeCSV)
	body, err := p.requestDocling(t.Context(), server.Client(), server.URL, "", map[string]any{"safe": true})
	if err != nil {
		t.Fatalf("requestDocling: %v", err)
	}
	if calls.Load() != 2 || !strings.Contains(string(body), `"chunks"`) {
		t.Fatalf("calls=%d body=%s", calls.Load(), body)
	}
}

func TestDoclingHybridRequestNeverEchoesErrorPayloadSecrets(t *testing.T) {
	t.Setenv("DOCLING_REQUEST_RETRY_MAX", "0")
	const secret = "Bearer super-secret-dgx-key"
	client := &http.Client{Transport: doclingRoundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusUnprocessableEntity,
			Body:       io.NopCloser(strings.NewReader(`{"detail":{"input":{"picture_description_api":{"headers":{"Authorization":"` + secret + `"}},"sources":[{"base64_string":"sensitive-source"}]}}}`)),
			Header:     make(http.Header),
		}, nil
	})}
	p := NewDoclingHybridParser(utility.FileTypePDF)
	_, err := p.requestDocling(t.Context(), client, "http://unit.test", secret, map[string]any{"base64_string": "sensitive-source"})
	if err == nil || !strings.Contains(err.Error(), "status 422") {
		t.Fatalf("error = %v, want status-only 422", err)
	}
	for _, forbidden := range []string{secret, "Authorization", "sensitive-source", "base64_string"} {
		if strings.Contains(err.Error(), forbidden) {
			t.Fatalf("error leaked %q: %v", forbidden, err)
		}
	}
}

func TestDoclingPinnedHTTPClientAllowsOnlyFixedOriginAndRejectsAllRedirects(t *testing.T) {
	t.Setenv("ALLOW_ANY_HOST", "true")
	previousLookup := common.LookupHost
	common.LookupHost = func(host string) ([]string, error) {
		if host != "docling-serve" {
			t.Fatalf("resolved host = %q, want docling-serve", host)
		}
		return []string{"10.89.0.12"}, nil
	}
	t.Cleanup(func() { common.LookupHost = previousLookup })

	for _, accepted := range []string{doclingTrustedOrigin, doclingTrustedOrigin + "/"} {
		client, err := doclingPinnedHTTPClient(accepted)
		if err != nil {
			t.Fatalf("trusted origin %q rejected: %v", accepted, err)
		}
		transport, ok := client.Transport.(*http.Transport)
		if !ok || transport.Proxy != nil {
			t.Fatalf("trusted client transport = %#v, want proxy disabled", client.Transport)
		}
		for _, target := range []string{
			"http://docling-serve:5001/next",
			"http://attacker.invalid/steal",
		} {
			req, _ := http.NewRequest(http.MethodGet, target, nil)
			if err := client.CheckRedirect(req, nil); err == nil {
				t.Fatalf("redirect to %q accepted", target)
			}
		}
	}

	for _, rejected := range []string{
		"https://docling-serve:5001",
		"http://docling-serve",
		"http://docling-serve:5002",
		"http://other:5001",
		"http://127.0.0.1:5001",
		"http://user:pass@docling-serve:5001",
		"http://docling-serve:5001//",
		"http://docling-serve:5001/v1",
		"http://docling-serve:5001?next=evil",
		"http://docling-serve:5001#fragment",
	} {
		if _, err := doclingPinnedHTTPClient(rejected); err == nil {
			t.Fatalf("untrusted Docling URL %q accepted", rejected)
		}
	}
}

func TestSharedDoclingHTTPClientReusesPinnedTransport(t *testing.T) {
	resetDoclingClientCache(t)
	previousLookup := common.LookupHost
	common.LookupHost = func(string) ([]string, error) { return []string{"10.89.0.12"}, nil }
	previousPinned := utility.PinnedHTTPClient
	var builds atomic.Int32
	utility.PinnedHTTPClient = func(string, string, time.Duration) *http.Client {
		builds.Add(1)
		return &http.Client{Transport: doclingRoundTripFunc(func(*http.Request) (*http.Response, error) {
			return nil, nil
		})}
	}
	t.Cleanup(func() {
		common.LookupHost = previousLookup
		utility.PinnedHTTPClient = previousPinned
	})

	first, err := sharedDoclingHTTPClient(doclingTrustedOrigin)
	if err != nil {
		t.Fatal(err)
	}
	second, err := sharedDoclingHTTPClient(doclingTrustedOrigin + "/")
	if err != nil {
		t.Fatal(err)
	}
	if first != second || builds.Load() != 1 {
		t.Fatalf("shared clients equal=%v builds=%d, want true/1", first == second, builds.Load())
	}
	if _, err := sharedDoclingHTTPClient("http://attacker.invalid:5001"); err == nil {
		t.Fatal("cached client bypassed fixed-origin validation")
	}
}

func TestDoclingNetworkFailureInvalidatesStalePinAndRepinsOnRetry(t *testing.T) {
	resetDoclingClientCache(t)
	t.Setenv("DOCLING_REQUEST_RETRY_MAX", "1")
	previousDelay := doclingRequestRetryDelay
	doclingRequestRetryDelay = time.Millisecond
	t.Cleanup(func() { doclingRequestRetryDelay = previousDelay })

	previousLookup := common.LookupHost
	var lookups atomic.Int32
	common.LookupHost = func(host string) ([]string, error) {
		if host != "docling-serve" {
			t.Fatalf("resolved host = %q, want docling-serve", host)
		}
		if lookups.Add(1) == 1 {
			return []string{"10.89.0.12"}, nil
		}
		return []string{"10.89.0.13"}, nil
	}
	previousPinned := utility.PinnedHTTPClient
	var builds atomic.Int32
	utility.PinnedHTTPClient = func(hostname, resolvedIP string, timeout time.Duration) *http.Client {
		if hostname != "docling-serve" || timeout != 0 {
			t.Fatalf("pinned client args = %q %q %s", hostname, resolvedIP, timeout)
		}
		builds.Add(1)
		return &http.Client{Transport: doclingRoundTripFunc(func(*http.Request) (*http.Response, error) {
			if resolvedIP == "10.89.0.12" {
				return nil, &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(`{"chunks":[],"documents":[],"processing_time":0}`)),
				Header:     make(http.Header),
			}, nil
		})}
	}
	t.Cleanup(func() {
		common.LookupHost = previousLookup
		utility.PinnedHTTPClient = previousPinned
	})

	stale, err := sharedDoclingHTTPClient(doclingTrustedOrigin)
	if err != nil {
		t.Fatal(err)
	}
	p := NewDoclingHybridParser(utility.FileTypeCSV)
	body, err := p.requestDocling(t.Context(), stale, doclingTrustedOrigin+"/v1/chunk/hybrid/source", "", map[string]any{"safe": true})
	if err != nil {
		t.Fatalf("requestDocling after repin: %v", err)
	}
	if !strings.Contains(string(body), `"chunks"`) {
		t.Fatalf("body = %s", body)
	}
	fresh, err := sharedDoclingHTTPClient(doclingTrustedOrigin)
	if err != nil {
		t.Fatal(err)
	}
	if fresh == stale {
		t.Fatal("network failure reused the stale pinned client")
	}
	if got := lookups.Load(); got != 2 {
		t.Fatalf("DNS lookups = %d, want initial pin plus one repin", got)
	}
	if got := builds.Load(); got != 2 {
		t.Fatalf("client builds = %d, want initial generation plus replacement", got)
	}
	if again, err := sharedDoclingHTTPClient(doclingTrustedOrigin); err != nil || again != fresh || builds.Load() != 2 {
		t.Fatalf("healthy replacement was not reused: again=%p fresh=%p builds=%d err=%v", again, fresh, builds.Load(), err)
	}
	// Concurrent late failures from the old generation must not evict the
	// healthy one or trigger redundant rebuilds.
	var wait sync.WaitGroup
	var staleEvictions atomic.Int32
	for range 16 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			if invalidateSharedDoclingHTTPClient(stale) {
				staleEvictions.Add(1)
			}
			if current, getErr := sharedDoclingHTTPClient(doclingTrustedOrigin); getErr != nil || current != fresh {
				staleEvictions.Add(1)
			}
		}()
	}
	wait.Wait()
	if staleEvictions.Load() != 0 {
		t.Fatalf("stale generation disturbed the replacement %d times", staleEvictions.Load())
	}
	if current, err := sharedDoclingHTTPClient(doclingTrustedOrigin); err != nil || current != fresh || builds.Load() != 2 {
		t.Fatalf("late stale invalidation replaced current client: current=%p fresh=%p builds=%d err=%v", current, fresh, builds.Load(), err)
	}
}

func TestDoclingRequestHasNoImplicitDeadlineAndCapsResponse(t *testing.T) {
	t.Setenv("DOCLING_REQUEST_RETRY_MAX", "0")
	t.Setenv("DOCLING_REQUEST_TIMEOUT_SECONDS", "0")
	t.Setenv("DOCLING_RESPONSE_MAX_BYTES", "8")
	var hadDeadline atomic.Bool
	client := &http.Client{Transport: doclingRoundTripFunc(func(req *http.Request) (*http.Response, error) {
		_, had := req.Context().Deadline()
		hadDeadline.Store(had)
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader("123456789")),
			Header:     make(http.Header),
		}, nil
	})}
	p := NewDoclingHybridParser(utility.FileTypeCSV)
	_, err := p.requestDocling(t.Context(), client, "http://unit.test", "", map[string]any{"safe": true})
	if err == nil || !strings.Contains(err.Error(), "DOCLING_RESPONSE_MAX_BYTES=8") {
		t.Fatalf("error = %v, want explicit response size failure", err)
	}
	if !errors.Is(err, errDoclingResponseTooLarge) {
		t.Fatalf("error = %v, want errDoclingResponseTooLarge", err)
	}
	if hadDeadline.Load() {
		t.Fatal("unset/zero request timeout installed an implicit per-attempt deadline")
	}
}

func TestDoclingHybridGenerationTextSupportsDescriptionOnlyImagesWithoutDuplication(t *testing.T) {
	const description = "압력 게이지가 0.42 MPa를 표시한다"
	if got := doclingHybridGenerationText("", []string{description}); got != description {
		t.Fatalf("description-only generation text = %q", got)
	}
	if got := doclingHybridGenerationText("화면: "+description, []string{description}); strings.Count(got, description) != 1 {
		t.Fatalf("description duplicated in %q", got)
	}
}

func TestDoclingPresentationPreviewItemsMarksNativeSources(t *testing.T) {
	previous := doclingPresentationParse
	doclingPresentationParse = func(context.Context, string, []byte) ParseResult {
		return ParseResult{
			JSON: MapsToItems([]map[string]any{
				{"text": "슬라이드 제목", DocTypeKey: DocTypeText, "slide_number": 1},
				{"text": "장비 사진", DocTypeKey: DocTypeImage, "image": "data:image/png;base64,aW1n", "slide_number": 1},
			}),
			Warnings: []string{"preview warning"},
		}
	}
	t.Cleanup(func() { doclingPresentationParse = previous })

	items, warnings, err := doclingPresentationPreviewItems(t.Context(), "deck.pptx", []byte("pptx"))
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 2 || len(warnings) != 1 {
		t.Fatalf("items=%d warnings=%#v", len(items), warnings)
	}
	for index, item := range ItemsToMaps(items) {
		if marked, _ := item[doclingPresentationPreviewSourceKey].(bool); !marked {
			t.Fatalf("item %d marker = %#v", index, item[doclingPresentationPreviewSourceKey])
		}
	}
}

func TestNativePresentationSemanticItemsAreSlideScopedAndBounded(t *testing.T) {
	long := strings.Repeat("가", nativePresentationSemanticMaxRunes+25)
	items, pageCount := nativePresentationSemanticItems([]map[string]any{
		{"text": "첫 슬라이드 제목", DocTypeKey: DocTypeText, "slide_number": 1},
		{"text": "", DocTypeKey: DocTypeImage, "image": "data:image/png;base64,aW1n", "slide_number": 1},
		{"text": long, DocTypeKey: DocTypeText, "slide_number": 2},
	})
	if pageCount != 2 {
		t.Fatalf("pageCount = %d, want 2", pageCount)
	}
	if len(items) != 3 {
		t.Fatalf("semantic items = %d, want one slide-1 and two bounded slide-2 chunks", len(items))
	}
	for index, item := range items {
		text, _ := item["text"].(string)
		if strings.TrimSpace(text) == "" || len([]rune(text)) > nativePresentationSemanticMaxRunes {
			t.Fatalf("item %d text runes = %d", index, len([]rune(text)))
		}
		if item["docling_native_semantic_fallback"] != true {
			t.Fatalf("item %d missing fallback marker: %#v", index, item)
		}
	}
	if got := items[0]["docling_page_numbers"]; len(got.([]int)) != 1 || got.([]int)[0] != 1 {
		t.Fatalf("slide-1 provenance = %#v", got)
	}
}

func TestEmptyDoclingPDFChunksUseMarkedNativeFallback(t *testing.T) {
	previous := doclingNativePDFParse
	doclingNativePDFParse = func(context.Context, string, []byte) ParseResult {
		return ParseResult{
			File: map[string]any{"page_count": 2},
			JSON: MapsToItems([]map[string]any{
				{"text": "ITM expiration settings", DocTypeKey: DocTypeText, "page_number": 1},
				{"text": "timestamp verification", DocTypeKey: DocTypeText, "page_number": 2},
			}),
			Warnings: []string{"native warning"},
		}
	}
	t.Cleanup(func() { doclingNativePDFParse = previous })

	result := NewDoclingHybridParser(utility.FileTypePDF).emptyPDFChunksFallback(t.Context(), "itm.pdf", []byte("pdf"))
	if result.Err != nil {
		t.Fatal(result.Err)
	}
	if result.File["docling_native_pdf_fallback"] != true || result.File["format"] != "pdf" {
		t.Fatalf("file metadata = %#v", result.File)
	}
	for index, item := range ItemsToMaps(result.JSON) {
		if marked, _ := item["docling_native_pdf_fallback"].(bool); !marked {
			t.Fatalf("item %d missing native PDF marker: %#v", index, item)
		}
	}
	if len(result.Warnings) != 2 || !strings.Contains(result.Warnings[1], "page previews") {
		t.Fatalf("warnings = %#v", result.Warnings)
	}
}

func TestEmptyDoclingPDFChunksRejectEmptyNativeFallback(t *testing.T) {
	previous := doclingNativePDFParse
	doclingNativePDFParse = func(context.Context, string, []byte) ParseResult {
		return ParseResult{JSON: MapsToItems([]map[string]any{{"text": ""}})}
	}
	t.Cleanup(func() { doclingNativePDFParse = previous })

	result := NewDoclingHybridParser(utility.FileTypePDF).emptyPDFChunksFallback(t.Context(), "empty.pdf", []byte("pdf"))
	if result.Err == nil || !strings.Contains(result.Err.Error(), "produced no semantic text") {
		t.Fatalf("error = %v", result.Err)
	}
}

func TestOversizedPresentationFallbackKeepsSemanticAndPreviewSources(t *testing.T) {
	previous := doclingPresentationParse
	doclingPresentationParse = func(context.Context, string, []byte) ParseResult {
		return ParseResult{
			JSON: MapsToItems([]map[string]any{
				{"text": "ALARM reset procedure", DocTypeKey: DocTypeText, "slide_number": 3},
				{"text": "HMI screenshot", DocTypeKey: DocTypeImage, "image": "data:image/png;base64,aW1n", "slide_number": 3},
			}),
			Warnings: []string{"native warning"},
		}
	}
	t.Cleanup(func() { doclingPresentationParse = previous })

	result := NewDoclingHybridParser(utility.FileTypePPTX).oversizedPresentationFallback(t.Context(), "huge.pptx", []byte("pptx"))
	if result.Err != nil {
		t.Fatal(result.Err)
	}
	if result.File["docling_native_semantic_fallback"] != true || result.File["page_count"] != 3 {
		t.Fatalf("file metadata = %#v", result.File)
	}
	maps := ItemsToMaps(result.JSON)
	semantic, preview := 0, 0
	for _, item := range maps {
		if marked, _ := item[doclingPresentationPreviewSourceKey].(bool); marked {
			preview++
		} else {
			semantic++
		}
	}
	if semantic != 1 || preview != 2 {
		t.Fatalf("semantic=%d preview=%d maps=%#v", semantic, preview, maps)
	}
	if len(result.Warnings) != 2 || !strings.Contains(result.Warnings[1], "bounded byte limit") {
		t.Fatalf("warnings = %#v", result.Warnings)
	}
}
