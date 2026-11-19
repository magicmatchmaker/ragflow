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

package parser

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"ragflow/internal/common"
	"ragflow/internal/entity"
	models "ragflow/internal/entity/models"
	"ragflow/internal/parser/tableutil"
	"ragflow/internal/utility"
)

const (
	defaultDoclingHybridTokenizer           = "Qwen/Qwen3-VL-Embedding-2B"
	defaultDoclingHybridMaxTokens           = 800
	defaultDoclingRequestRetryMax           = 3
	defaultDoclingResponseMaxBytes          = 256 << 20
	defaultDoclingPictureDescriptionTimeout = 86400
	defaultDoclingPictureDescriptionWorkers = 2
	defaultDoclingPictureDescriptionArea    = 0.0
	defaultDoclingPictureDescriptionPrompt  = "Describe this image faithfully for technical retrieval. Preserve visible labels, values, states, warnings, diagram relationships, and table contents. Use the document language and do not infer facts that are not visible."
	doclingTrustedOrigin                    = "http://docling-serve:5001"
	nativePresentationSemanticMaxRunes      = 6000
)

var errDoclingResponseTooLarge = errors.New("Docling response exceeds configured byte limit")

// Test seam only; production keeps the shared bounded exponential backoff.
var doclingRequestRetryDelay = common.DefaultRetryDelay

// doclingPresentationParse keeps PPTX preview generation on RAGFlow's native
// Go presentation path. Docling remains authoritative for semantic text
// boundaries, while the native parser supplies slide geometry and embedded
// media needed to render a faithful full-slide preview. The seam is overridden
// only by unit tests; production always calls PPTXParser.
var doclingPresentationParse = func(ctx context.Context, filename string, data []byte) ParseResult {
	return NewPPTXParser().ParseWithResult(ctx, filename, data)
}

// doclingNativePDFParse is the bounded recovery path for a valid PDF that
// Docling converts successfully but does not chunk. Prefer the deterministic
// text layer first (it is both cheaper and more faithful for print-to-PDF
// manuals), then use the native DeepDoc OCR/layout parser only when the text
// layer itself is empty. The seam is overridden only by unit tests.
var doclingNativePDFParse = func(ctx context.Context, filename string, data []byte) ParseResult {
	plain := NewPDFParser()
	plain.ParseMethod = "plain_text"
	result := plain.ParseWithResult(ctx, filename, data)
	if result.Err == nil && nativeItemsContainSemanticText(ItemsToMaps(result.JSON)) {
		return result
	}
	return NewPDFParser().ParseWithResult(ctx, filename, data)
}

const doclingPresentationPreviewSourceKey = "docling_presentation_preview_source"

var doclingHTTPClientCache struct {
	sync.Mutex
	client *http.Client
}

// DoclingHybridParser is the thin adapter for Docling Serve's official
// /v1/chunk/hybrid/source endpoint. Docling owns conversion and chunk
// boundaries; this adapter only projects the returned chunks onto RAGFlow's
// existing DocItem contract and resolves doc_items JSON pointers back to the
// full converted document for page, bbox, and image provenance.
type DoclingHybridParser struct {
	fileType utility.FileType
}

func NewDoclingHybridParser(fileType utility.FileType) *DoclingHybridParser {
	return &DoclingHybridParser{fileType: fileType}
}

// ConfigureFromSetup intentionally ignores Docling operational settings.
// Endpoints, models, tokenizer choices and credentials are deployment-owned
// environment configuration; accepting them from a document parser_config
// would allow SSRF and picture-service credential exfiltration.
func (p *DoclingHybridParser) ConfigureFromSetup(_ map[string]any) {}

type doclingHybridChunk struct {
	Filename    string         `json:"filename"`
	ChunkIndex  int            `json:"chunk_index"`
	Text        string         `json:"text"`
	RawText     *string        `json:"raw_text"`
	NumTokens   *int           `json:"num_tokens"`
	Headings    []string       `json:"headings"`
	Captions    []string       `json:"captions"`
	DocItems    []string       `json:"doc_items"`
	PageNumbers []int          `json:"page_numbers"`
	Metadata    map[string]any `json:"metadata"`
}

type doclingHybridExportDocument struct {
	Filename    string         `json:"filename"`
	JSONContent map[string]any `json:"json_content"`
}

type doclingHybridDocumentResult struct {
	// Docling Serve 1.38.2's live wire schema names the export payload
	// "content". Document is retained only as a compatibility reader for
	// earlier builds; new requests/tests use Content.
	Content  doclingHybridExportDocument `json:"content"`
	Document doclingHybridExportDocument `json:"document,omitempty"`
	Status   string                      `json:"status"`
	Errors   []doclingHybridError        `json:"errors"`
}

type doclingHybridError struct {
	ComponentType string `json:"component_type"`
	ModuleName    string `json:"module_name"`
	ErrorMessage  string `json:"error_message"`
	Category      string `json:"category"`
	PageNo        *int   `json:"page_no"`
}

type doclingHybridResponse struct {
	Chunks         []doclingHybridChunk          `json:"chunks"`
	Documents      []doclingHybridDocumentResult `json:"documents"`
	ProcessingTime float64                       `json:"processing_time"`
}

func (p *DoclingHybridParser) ParseWithResult(ctx context.Context, filename string, data []byte) ParseResult {
	if len(data) == 0 {
		return ParseResult{Err: fmt.Errorf("parser: Docling Hybrid source %q is empty", filename)}
	}
	// The Docling endpoint is an operator-owned deployment setting, not a
	// document/parser option. Keeping it out of parser_config makes the exact
	// configured origin the private-network allowlist and prevents a document
	// owner from turning this adapter into an SSRF primitive.
	serverURL := strings.TrimSpace(common.GetEnv(common.EnvDoclingServerURL))
	if serverURL == "" {
		return ParseResult{Err: fmt.Errorf("parser: Docling Hybrid requires %s", common.EnvDoclingServerURL)}
	}
	auth := ""
	if apiKey := strings.TrimSpace(common.GetEnv(common.EnvDoclingAPIKey)); apiKey != "" {
		auth = "Bearer " + apiKey
	}
	payload, payloadErr := p.requestPayload(filename, data)
	if payloadErr != nil {
		return ParseResult{Err: fmt.Errorf("parser: Docling Hybrid request options: %w", payloadErr)}
	}
	client, err := sharedDoclingHTTPClient(serverURL)
	if err != nil {
		return ParseResult{Err: fmt.Errorf("parser: Docling Hybrid endpoint: %w", err)}
	}
	body, err := p.requestDocling(ctx, client, strings.TrimSuffix(serverURL, "/")+"/v1/chunk/hybrid/source", auth, payload)
	if err != nil {
		if p.fileType == utility.FileTypePPTX && errors.Is(err, errDoclingResponseTooLarge) {
			return p.oversizedPresentationFallback(ctx, filename, data)
		}
		return ParseResult{Err: fmt.Errorf("parser: Docling Hybrid request: %w", err)}
	}
	var decoded doclingHybridResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		return ParseResult{Err: fmt.Errorf("parser: Docling Hybrid decode response: %w", err)}
	}
	if len(decoded.Documents) != 1 {
		return ParseResult{Err: fmt.Errorf("parser: Docling Hybrid returned %d document results for one source", len(decoded.Documents))}
	}
	if err := validateDoclingHybridDocument(decoded.Documents[0]); err != nil {
		return ParseResult{Err: fmt.Errorf("parser: Docling Hybrid conversion: %w", err)}
	}
	exported := decoded.Documents[0].Content
	if len(exported.JSONContent) == 0 {
		exported = decoded.Documents[0].Document
	}
	if len(exported.JSONContent) == 0 {
		return ParseResult{Err: fmt.Errorf("parser: Docling Hybrid response omitted converted document JSON; include_converted_doc=true is required")}
	}
	if len(decoded.Chunks) == 0 && p.fileType == utility.FileTypePDF {
		return p.emptyPDFChunksFallback(ctx, filename, data)
	}
	if len(decoded.Chunks) == 0 && p.fileType != utility.FileTypePPTX {
		return ParseResult{Err: fmt.Errorf("parser: Docling Hybrid response contained no chunks")}
	}
	content := exported.JSONContent
	items := []DocItem(nil)
	if len(decoded.Chunks) > 0 {
		items, err = doclingHybridChunksToItems(decoded.Chunks, content)
		if err != nil {
			return ParseResult{Err: fmt.Errorf("parser: Docling Hybrid provenance: %w", err)}
		}
	}
	warnings := []string(nil)
	previewSourceCount := 0
	if p.fileType == utility.FileTypePPTX {
		if len(decoded.Chunks) == 0 {
			warnings = append(warnings, "Docling returned no semantic chunks; indexed native full-slide previews only")
		}
		previewItems, previewWarnings, previewErr := doclingPresentationPreviewItems(ctx, filename, data)
		if previewErr != nil {
			return ParseResult{Err: fmt.Errorf("parser: Docling Hybrid presentation preview source: %w", previewErr)}
		}
		previewSourceCount = len(previewItems)
		items = append(items, previewItems...)
		warnings = append(warnings, previewWarnings...)
	}
	return ParseResult{
		File: map[string]any{
			"name":                         filename,
			"format":                       string(p.fileType),
			"page_count":                   doclingHybridPageCount(content),
			"docling_processing_time":      decoded.ProcessingTime,
			"presentation_preview_sources": previewSourceCount,
		},
		JSON:     items,
		Warnings: warnings,
	}
}

// emptyPDFChunksFallback retains a successfully converted PDF when Docling's
// HybridChunker returns no boundaries. Every native record is marked so the
// downstream hybrid chunker can merge it page-by-page, render the original
// PDF page, run local OCR where needed, and keep the page preview linked to the
// contextualized text. This is a recovery path, not an alternate default.
func (p *DoclingHybridParser) emptyPDFChunksFallback(ctx context.Context, filename string, data []byte) ParseResult {
	parsed := doclingNativePDFParse(ctx, filename, data)
	if parsed.Err != nil {
		return ParseResult{Err: fmt.Errorf("parser: Docling Hybrid empty-chunk native PDF fallback: %w", parsed.Err)}
	}
	maps := ItemsToMaps(parsed.JSON)
	if !nativeItemsContainSemanticText(maps) {
		return ParseResult{Err: errors.New("parser: Docling Hybrid empty-chunk native PDF fallback produced no semantic text")}
	}
	for _, item := range maps {
		item["docling_native_pdf_fallback"] = true
	}
	file := make(map[string]any, len(parsed.File)+3)
	for key, value := range parsed.File {
		file[key] = value
	}
	file["name"] = filename
	file["format"] = string(p.fileType)
	file["docling_native_pdf_fallback"] = true
	warnings := append([]string(nil), parsed.Warnings...)
	warnings = append(warnings, "Docling returned no hybrid chunks; used native PDF text/OCR fallback with page previews")
	return ParseResult{File: file, JSON: MapsToItems(maps), Warnings: warnings}
}

func nativeItemsContainSemanticText(items []map[string]any) bool {
	for _, item := range items {
		if strings.TrimSpace(nativePresentationSemanticText(item)) != "" {
			return true
		}
	}
	return false
}

// oversizedPresentationFallback keeps very large decks in the contextual
// pipeline when Docling's converted-document JSON alone exceeds the bounded
// response limit. The native office parser reads the PPTX once, its visible
// text and tables are collapsed into bounded slide-scoped semantic chunks,
// and the same native records remain private inputs for full-slide previews.
// This is deliberately PPTX-only: unlike PDFs, presentations already have a
// deterministic native slide model and preview renderer in this process.
func (p *DoclingHybridParser) oversizedPresentationFallback(ctx context.Context, filename string, data []byte) ParseResult {
	parsed := doclingPresentationParse(ctx, filename, data)
	if parsed.Err != nil {
		return ParseResult{Err: fmt.Errorf("parser: oversized presentation native fallback: %w", parsed.Err)}
	}
	maps := ItemsToMaps(parsed.JSON)
	semantic, pageCount := nativePresentationSemanticItems(maps)
	if len(semantic) == 0 {
		return ParseResult{Err: errors.New("parser: oversized presentation native fallback produced no semantic text")}
	}
	preview := make([]map[string]any, 0, len(maps))
	for _, source := range maps {
		item := make(map[string]any, len(source)+1)
		for key, value := range source {
			item[key] = value
		}
		item[doclingPresentationPreviewSourceKey] = true
		preview = append(preview, item)
	}
	semantic = append(semantic, preview...)
	warnings := append([]string(nil), parsed.Warnings...)
	warnings = append(warnings, "Docling converted response exceeded the bounded byte limit; used native slide-scoped semantic fallback")
	return ParseResult{
		File: map[string]any{
			"name":                             filename,
			"format":                           string(p.fileType),
			"page_count":                       pageCount,
			"presentation_preview_sources":     len(preview),
			"docling_native_semantic_fallback": true,
			"docling_native_semantic_fallback_reason": "response_too_large",
		},
		JSON:     MapsToItems(semantic),
		Warnings: warnings,
	}
}

func nativePresentationSemanticItems(items []map[string]any) ([]map[string]any, int) {
	byPage := make(map[int][]string)
	order := make([]int, 0)
	seen := make(map[int]struct{})
	pageCount := 0
	for _, item := range items {
		page := nativePresentationPageNumber(item)
		if page <= 0 {
			continue
		}
		if page > pageCount {
			pageCount = page
		}
		text := strings.TrimSpace(nativePresentationSemanticText(item))
		if text == "" {
			continue
		}
		if _, ok := seen[page]; !ok {
			seen[page] = struct{}{}
			order = append(order, page)
		}
		byPage[page] = append(byPage[page], text)
	}
	out := make([]map[string]any, 0, len(order))
	for _, page := range order {
		for _, text := range splitNativePresentationSemanticText(strings.Join(byPage[page], "\n\n"), nativePresentationSemanticMaxRunes) {
			out = append(out, map[string]any{
				"text":                             text,
				DocTypeKey:                         DocTypeText,
				"page_number":                      page,
				"slide_number":                     page,
				"docling_page_numbers":             []int{page},
				"docling_native_semantic_fallback": true,
			})
		}
	}
	return out, pageCount
}

func nativePresentationSemanticText(item map[string]any) string {
	if text, _ := item["text"].(string); strings.TrimSpace(text) != "" {
		return text
	}
	if table, ok := item["table"].(*entity.TableData); ok && table != nil {
		return tableutil.RenderTableHTML(table)
	}
	return ""
}

func nativePresentationPageNumber(item map[string]any) int {
	switch value := item["slide_number"].(type) {
	case int:
		return value
	case int64:
		return int(value)
	case float64:
		return int(value)
	case json.Number:
		page, _ := strconv.Atoi(value.String())
		return page
	default:
		return 0
	}
}

func splitNativePresentationSemanticText(text string, maxRunes int) []string {
	text = strings.TrimSpace(text)
	if text == "" || maxRunes <= 0 {
		return nil
	}
	runes := []rune(text)
	parts := make([]string, 0, len(runes)/maxRunes+1)
	for len(runes) > 0 {
		cut := len(runes)
		if cut > maxRunes {
			cut = maxRunes
			// Prefer a nearby line boundary without producing a tiny prefix.
			for index := cut - 1; index >= cut/2; index-- {
				if runes[index] == '\n' {
					cut = index + 1
					break
				}
			}
		}
		if part := strings.TrimSpace(string(runes[:cut])); part != "" {
			parts = append(parts, part)
		}
		runes = runes[cut:]
	}
	return parts
}

// doclingPresentationPreviewItems parses the same PPTX bytes with RAGFlow's
// built-in office_oxide backend and marks the resulting geometry/media items
// as private preview sources. DoclingHybridChunker consumes the marker and
// turns these records into one full-slide image chunk per slide; the marked
// source records never reach the index themselves.
func doclingPresentationPreviewItems(ctx context.Context, filename string, data []byte) ([]DocItem, []string, error) {
	result := doclingPresentationParse(ctx, filename, data)
	if result.Err != nil {
		return nil, nil, result.Err
	}
	maps := ItemsToMaps(result.JSON)
	if len(maps) == 0 {
		return nil, nil, errors.New("native presentation parser produced no slide records")
	}
	for _, item := range maps {
		item[doclingPresentationPreviewSourceKey] = true
	}
	return MapsToItems(maps), result.Warnings, nil
}

func (p *DoclingHybridParser) requestDocling(ctx context.Context, client *http.Client, endpoint, auth string, payload map[string]any) ([]byte, error) {
	// Zero/unset intentionally means task-context only. Operators may opt into
	// a per-attempt network deadline without creating a whole-document cutoff.
	timeoutSeconds := common.GetEnvInt(common.EnvDoclingRequestTimeout, 0)
	retryMax := common.GetEnvInt(common.EnvDoclingRequestRetryMax, defaultDoclingRequestRetryMax)
	if retryMax < 0 {
		retryMax = defaultDoclingRequestRetryMax
	}
	activeClient := client
	sharedManaged := isSharedDoclingHTTPClient(client) || endpoint == doclingTrustedOrigin+"/v1/chunk/hybrid/source"
	var responseBody []byte
	err := common.RetryWithBackoff(ctx, retryMax, doclingRequestRetryDelay, func() error {
		if activeClient == nil {
			refreshed, refreshErr := sharedDoclingHTTPClient(doclingTrustedOrigin)
			if refreshErr != nil {
				return fmt.Errorf("refresh pinned Docling client: %w", refreshErr)
			}
			activeClient = refreshed
		}
		attemptCtx := ctx
		cancel := func() {}
		if timeoutSeconds > 0 {
			attemptCtx, cancel = context.WithTimeout(ctx, time.Duration(timeoutSeconds)*time.Second)
		}
		defer cancel()
		resp, err := models.PostJSONRequest(attemptCtx, activeClient, endpoint, auth, payload)
		if err != nil {
			if sharedManaged && isDoclingNetworkError(err) {
				// A restarted container may keep the same DNS name while receiving
				// a new IP. Clear only the exact failed generation: another request
				// may already have installed a fresh pinned client, and an older
				// in-flight failure must never evict it. The next retry resolves and
				// pins the hostname again before issuing the request.
				invalidateSharedDoclingHTTPClient(activeClient)
				activeClient = nil
			}
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			const maxErrorBody = 4096
			// FastAPI validation errors can echo the submitted source and nested
			// picture_description_api headers. Drain only a bounded prefix and
			// report status-only so Bearer secrets/base64 never reach task errors,
			// DB state, or logs.
			_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxErrorBody+1))
			return fmt.Errorf("Docling HTTP status %d", resp.StatusCode)
		}
		maxBytes := int64(common.GetEnvInt(common.EnvDoclingResponseMaxBytes, defaultDoclingResponseMaxBytes))
		if maxBytes <= 0 {
			maxBytes = defaultDoclingResponseMaxBytes
		}
		body, readErr := io.ReadAll(io.LimitReader(resp.Body, maxBytes+1))
		if readErr != nil {
			return fmt.Errorf("read Docling response: %w", readErr)
		}
		if int64(len(body)) > maxBytes {
			return fmt.Errorf("%w: %s=%d bytes", errDoclingResponseTooLarge, common.EnvDoclingResponseMaxBytes, maxBytes)
		}
		responseBody = body
		return nil
	}, func(err error) bool {
		return !errors.Is(err, errDoclingResponseTooLarge) && common.IsTransientError(err)
	})
	if err != nil {
		return nil, err
	}
	return responseBody, nil
}

func isDoclingNetworkError(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) {
		return false
	}
	var urlErr *url.Error
	if errors.As(err, &urlErr) {
		return true
	}
	var netErr net.Error
	return errors.As(err, &netErr)
}

func isSharedDoclingHTTPClient(client *http.Client) bool {
	if client == nil {
		return false
	}
	doclingHTTPClientCache.Lock()
	defer doclingHTTPClientCache.Unlock()
	return doclingHTTPClientCache.client == client
}

// invalidateSharedDoclingHTTPClient removes a failed pinned-client generation
// only when it is still current. Pointer comparison is the generation token:
// an old request cannot erase the replacement installed by a newer request.
func invalidateSharedDoclingHTTPClient(stale *http.Client) bool {
	if stale == nil {
		return false
	}
	doclingHTTPClientCache.Lock()
	if doclingHTTPClientCache.client != stale {
		doclingHTTPClientCache.Unlock()
		return false
	}
	doclingHTTPClientCache.client = nil
	doclingHTTPClientCache.Unlock()
	stale.CloseIdleConnections()
	return true
}

func sharedDoclingHTTPClient(rawURL string) (*http.Client, error) {
	if _, err := validateDoclingOrigin(rawURL); err != nil {
		return nil, err
	}
	doclingHTTPClientCache.Lock()
	defer doclingHTTPClientCache.Unlock()
	if doclingHTTPClientCache.client != nil {
		return doclingHTTPClientCache.client, nil
	}
	client, err := doclingPinnedHTTPClient(rawURL)
	if err != nil {
		return nil, err
	}
	doclingHTTPClientCache.client = client
	return client, nil
}

// doclingPinnedHTTPClient permits only the fixed production Docling origin,
// pins the one DNS resolution used for the connection and rejects every
// redirect. It intentionally ignores ALLOW_ANY_HOST and environment proxies.
// The per-attempt context in requestDocling bounds network hangs without
// imposing a separate whole-document deadline.
func doclingPinnedHTTPClient(rawURL string) (*http.Client, error) {
	parsed, err := validateDoclingOrigin(rawURL)
	if err != nil {
		return nil, err
	}
	hostname := parsed.Hostname()
	if net.ParseIP(hostname) != nil {
		return nil, fmt.Errorf("IP literals are not allowed in Docling URL")
	}
	resolvedIP, err := common.ResolveHostPin(hostname)
	if err != nil {
		return nil, fmt.Errorf("resolve configured Docling host %q: %w", hostname, err)
	}
	client := utility.PinnedHTTPClient(hostname, resolvedIP, 0)
	client.CheckRedirect = func(req *http.Request, _ []*http.Request) error {
		return fmt.Errorf("Docling redirects are disabled: %s", req.URL.Redacted())
	}
	return client, nil
}

func validateDoclingOrigin(rawURL string) (*url.URL, error) {
	parsed, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil {
		return nil, fmt.Errorf("invalid URL: %w", err)
	}
	if parsed.User != nil {
		return nil, fmt.Errorf("userinfo is not allowed in Docling URL")
	}
	if parsed.Scheme != "http" || !strings.EqualFold(parsed.Hostname(), "docling-serve") || parsed.Port() != "5001" {
		return nil, fmt.Errorf("Docling URL must use the trusted origin %s", doclingTrustedOrigin)
	}
	if parsed.Opaque != "" || (parsed.Path != "" && parsed.Path != "/") || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, fmt.Errorf("Docling URL must be an origin without path, query, or fragment")
	}
	return parsed, nil
}

func validateDoclingHybridDocument(document doclingHybridDocumentResult) error {
	status := strings.ToLower(strings.TrimSpace(document.Status))
	if status != "success" {
		return fmt.Errorf("status=%q%s", status, doclingHybridErrorSummary(document.Errors))
	}
	if len(document.Errors) > 0 {
		return fmt.Errorf("status=success included conversion errors%s", doclingHybridErrorSummary(document.Errors))
	}
	return nil
}

func doclingHybridErrorSummary(errors []doclingHybridError) string {
	if len(errors) == 0 {
		return ""
	}
	parts := make([]string, 0, len(errors))
	for _, item := range errors {
		// Docling modules can forward remote-service error strings. Do not copy
		// those strings into durable task errors because they may echo request
		// headers, API keys, or embedded source bytes.
		message := strings.TrimSpace(item.Category)
		if message == "" {
			message = "conversion error"
		}
		where := strings.TrimSpace(item.ModuleName)
		if where != "" {
			message = where + ": " + message
		}
		if item.PageNo != nil {
			message = fmt.Sprintf("page %d: %s", *item.PageNo, message)
		}
		parts = append(parts, message)
	}
	return ": " + strings.Join(parts, "; ")
}

func (p *DoclingHybridParser) requestPayload(filename string, data []byte) (map[string]any, error) {
	format := doclingHybridInputFormat(p.fileType)
	includeImages := p.fileType != utility.FileTypePPTX
	imageExportMode := "embedded"
	if !includeImages {
		// PPTX media is rendered faithfully by the native full-slide preview
		// path below. Re-embedding every source image in Docling's converted JSON
		// can exceed hundreds of MiB without adding a second useful preview.
		imageExportMode = "placeholder"
	}
	convertOptions := map[string]any{
		"from_formats":      []string{format},
		"to_formats":        []string{"json"},
		"include_images":    includeImages,
		"image_export_mode": imageExportMode,
	}
	if value, ok := doclingEnvBool(common.EnvDoclingDoOCR); ok {
		convertOptions["do_ocr"] = value
	}
	if backend := strings.TrimSpace(common.GetEnv(common.EnvDoclingPDFBackend)); backend != "" && p.fileType == utility.FileTypePDF {
		convertOptions["pdf_backend"] = backend
	}
	if pictureURL := strings.TrimSpace(common.GetEnv(common.EnvDoclingPictureDescriptionURL)); pictureURL != "" {
		pictureArea := defaultDoclingPictureDescriptionArea
		if raw := strings.TrimSpace(common.GetEnv(common.EnvDoclingPictureDescriptionArea)); raw != "" {
			parsed, err := strconv.ParseFloat(raw, 64)
			if err != nil || parsed < 0 || parsed > 1 {
				return nil, fmt.Errorf("%s must be a number between 0 and 1", common.EnvDoclingPictureDescriptionArea)
			}
			pictureArea = parsed
		}
		pictureTimeout := common.GetEnvInt(common.EnvDoclingPictureDescriptionTimeout, defaultDoclingPictureDescriptionTimeout)
		if pictureTimeout <= 0 {
			pictureTimeout = defaultDoclingPictureDescriptionTimeout
		}
		pictureWorkers := common.GetEnvInt(common.EnvDoclingPictureDescriptionWorkers, defaultDoclingPictureDescriptionWorkers)
		if pictureWorkers <= 0 {
			pictureWorkers = defaultDoclingPictureDescriptionWorkers
		}
		picturePrompt := strings.TrimSpace(common.GetEnv(common.EnvDoclingPictureDescriptionPrompt))
		if picturePrompt == "" {
			picturePrompt = defaultDoclingPictureDescriptionPrompt
		}
		api := map[string]any{
			"url":         pictureURL,
			"timeout":     pictureTimeout,
			"concurrency": pictureWorkers,
			"prompt":      picturePrompt,
		}
		key, err := p.pictureDescriptionAPIKey()
		if err != nil {
			return nil, err
		}
		if key != "" {
			api["headers"] = map[string]string{"Authorization": "Bearer " + key}
		}
		if model := strings.TrimSpace(common.GetEnv(common.EnvDoclingPictureDescriptionModel)); model != "" {
			api["params"] = map[string]any{"model": model, "temperature": 0, "max_tokens": 320}
		}
		convertOptions["do_picture_description"] = true
		convertOptions["picture_description_area_threshold"] = pictureArea
		convertOptions["picture_description_api"] = api
	}
	maxTokens := common.GetEnvInt(common.EnvDoclingHybridMaxTokens, defaultDoclingHybridMaxTokens)
	if maxTokens <= 0 {
		maxTokens = defaultDoclingHybridMaxTokens
	}
	tokenizerName := strings.TrimSpace(common.GetEnv(common.EnvDoclingHybridTokenizer))
	if tokenizerName == "" {
		tokenizerName = defaultDoclingHybridTokenizer
	}
	return map[string]any{
		"convert_options": convertOptions,
		"sources": []map[string]any{{
			"kind":          "file",
			"filename":      filename,
			"base64_string": base64.StdEncoding.EncodeToString(data),
		}},
		"include_converted_doc": true,
		"chunking_options": map[string]any{
			"chunker":             "hybrid",
			"include_raw_text":    true,
			"max_tokens":          maxTokens,
			"tokenizer":           tokenizerName,
			"merge_peers":         true,
			"use_markdown_tables": true,
			"use_markdown_images": false,
		},
	}, nil
}

func (p *DoclingHybridParser) pictureDescriptionAPIKey() (string, error) {
	// Secrets are file-only: parser_config and plaintext environment values are
	// deliberately not accepted because both can leak through dataset/config
	// inspection. Production mounts the file read-only.
	return common.GetEnvSecret("", common.EnvDoclingPictureDescriptionAPIKeyFile)
}

func doclingHybridInputFormat(fileType utility.FileType) string {
	switch fileType {
	case utility.FileTypeVISUAL:
		return "image"
	case utility.FileTypeMarkdown:
		return "md"
	default:
		return string(fileType)
	}
}

func doclingEnvBool(envKey string) (bool, bool) {
	raw := strings.TrimSpace(common.GetEnv(envKey))
	if raw == "" {
		return false, false
	}
	value, err := strconv.ParseBool(raw)
	return value, err == nil
}

func doclingHybridChunksToItems(chunks []doclingHybridChunk, content map[string]any) ([]DocItem, error) {
	heights := doclingPageHeights(content)
	items := make([]map[string]any, 0, len(chunks))
	for _, chunk := range chunks {
		text := strings.TrimSpace(chunk.Text)
		if text == "" && chunk.RawText != nil && strings.TrimSpace(*chunk.RawText) != "" {
			text = strings.TrimSpace(*chunk.RawText)
		}
		item := map[string]any{
			DocTypeKey:            DocTypeText,
			"docling_doc_items":   append([]string(nil), chunk.DocItems...),
			"docling_chunk_index": chunk.ChunkIndex,
		}
		if chunk.RawText != nil {
			item["docling_raw_text"] = *chunk.RawText
		}
		if chunk.NumTokens != nil {
			item["docling_num_tokens"] = *chunk.NumTokens
		}
		if len(chunk.Headings) > 0 {
			item["docling_headings"] = append([]string(nil), chunk.Headings...)
		}
		if len(chunk.Captions) > 0 {
			item["docling_captions"] = append([]string(nil), chunk.Captions...)
		}
		if len(chunk.PageNumbers) > 0 {
			item["docling_page_numbers"] = append([]int(nil), chunk.PageNumbers...)
			item["page_number"] = chunk.PageNumbers[0]
		}
		if len(chunk.Metadata) > 0 {
			item["docling_metadata"] = chunk.Metadata
		}

		var positions [][]any
		var pictureRefs []string
		var pictureMedia []map[string]any
		var descriptions []string
		for _, ref := range chunk.DocItems {
			resolved, err := doclingResolveJSONPointer(content, ref)
			if err != nil {
				return nil, fmt.Errorf("chunk %d doc_item %q: %w", chunk.ChunkIndex, ref, err)
			}
			entry, ok := resolved.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("chunk %d doc_item %q is not an object", chunk.ChunkIndex, ref)
			}
			var entryPositions [][]any
			for _, prov := range doclingProvs(entry) {
				if pos := doclingProvPosition(prov, heights); pos != nil {
					if !doclingPositionExists(entryPositions, pos) {
						entryPositions = append(entryPositions, pos)
					}
					if !doclingPositionExists(positions, pos) {
						positions = append(positions, pos)
					}
				}
			}
			if doclingHybridIsPicture(ref, entry) {
				pictureRefs = append(pictureRefs, ref)
				entryDescriptions := doclingHybridDescriptions(entry)
				if image := doclingHybridImageURI(entry); image != "" {
					media := map[string]any{"ref": ref, "image": image}
					if len(entryPositions) > 0 {
						media["positions"] = entryPositions
					}
					if len(entryDescriptions) > 0 {
						media["descriptions"] = entryDescriptions
					}
					pictureMedia = append(pictureMedia, media)
					if _, exists := item["image"]; !exists {
						item["image"] = image
						item[DocTypeKey] = DocTypeImage
					}
				}
				descriptions = append(descriptions, entryDescriptions...)
			}
		}
		if len(positions) > 0 {
			item["positions"] = positions
			item["_pdf_positions"] = positions
		}
		if len(pictureRefs) > 0 {
			item["docling_picture_refs"] = pictureRefs
		}
		if len(pictureMedia) > 0 {
			// The typed chunk schema has one primary image. Keep complete,
			// per-picture provenance until DoclingHybridChunker expands it into
			// independently uploaded media chunks; the internal field is removed
			// before indexing.
			item["docling_picture_media"] = pictureMedia
		}
		if len(descriptions) > 0 {
			item["docling_picture_descriptions"] = descriptions
		}
		// HybridChunker text is the primary generation content. Some Docling
		// serializers keep picture enrichment only in PictureItem.meta instead
		// of repeating it in chunk.text, so append every missing description.
		// This also gives standalone image documents searchable/generation text
		// when their HybridChunk has no OCR text of its own.
		text = doclingHybridGenerationText(text, descriptions)
		if text == "" {
			continue
		}
		item["text"] = text
		items = append(items, item)
	}
	if len(items) == 0 {
		return nil, fmt.Errorf("no non-empty chunks")
	}
	return MapsToItems(items), nil
}

func doclingHybridGenerationText(text string, descriptions []string) string {
	text = strings.TrimSpace(text)
	missing := make([]string, 0, len(descriptions))
	for _, description := range descriptions {
		description = strings.TrimSpace(description)
		if description != "" && !strings.Contains(text, description) {
			missing = append(missing, description)
		}
	}
	if len(missing) == 0 {
		return text
	}
	descriptionText := strings.Join(missing, "\n")
	if text == "" {
		return descriptionText
	}
	return text + "\n\n[Picture description]\n" + descriptionText
}

func doclingResolveJSONPointer(root any, pointer string) (any, error) {
	if pointer == "#" || pointer == "" {
		return root, nil
	}
	if !strings.HasPrefix(pointer, "#/") {
		return nil, fmt.Errorf("unsupported JSON pointer")
	}
	current := root
	for _, encoded := range strings.Split(strings.TrimPrefix(pointer, "#/"), "/") {
		part := strings.ReplaceAll(strings.ReplaceAll(encoded, "~1", "/"), "~0", "~")
		switch node := current.(type) {
		case map[string]any:
			value, ok := node[part]
			if !ok {
				return nil, fmt.Errorf("missing segment %q", part)
			}
			current = value
		case []any:
			index, err := strconv.Atoi(part)
			if err != nil || index < 0 || index >= len(node) {
				return nil, fmt.Errorf("invalid array segment %q", part)
			}
			current = node[index]
		default:
			return nil, fmt.Errorf("segment %q traverses a scalar", part)
		}
	}
	return current, nil
}

func doclingHybridIsPicture(ref string, entry map[string]any) bool {
	if strings.HasPrefix(ref, "#/pictures/") {
		return true
	}
	label, _ := entry["label"].(string)
	return strings.EqualFold(label, "picture") || strings.EqualFold(label, "chart")
}

func doclingHybridImageURI(entry map[string]any) string {
	switch image := entry["image"].(type) {
	case string:
		return strings.TrimSpace(image)
	case map[string]any:
		uri, _ := image["uri"].(string)
		return strings.TrimSpace(uri)
	default:
		return ""
	}
}

func doclingHybridDescriptions(entry map[string]any) []string {
	var out []string
	// Current Docling core stores picture enrichment under
	// PictureItem.meta.description (DescriptionMetaField).
	if meta, ok := entry["meta"].(map[string]any); ok {
		if description, ok := meta["description"].(map[string]any); ok {
			if text, _ := description["text"].(string); strings.TrimSpace(text) != "" {
				out = append(out, strings.TrimSpace(text))
			}
		}
	}
	// Older Docling documents exposed the same enrichment as a description
	// annotation. Keep reading it so existing converted JSON remains usable.
	annotations, _ := entry["annotations"].([]any)
	for _, raw := range annotations {
		annotation, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		kind, _ := annotation["kind"].(string)
		text, _ := annotation["text"].(string)
		if strings.EqualFold(kind, "description") && strings.TrimSpace(text) != "" {
			out = append(out, strings.TrimSpace(text))
		}
	}
	return out
}

func doclingPositionExists(positions [][]any, candidate []any) bool {
	want, _ := json.Marshal(candidate)
	for _, existing := range positions {
		got, _ := json.Marshal(existing)
		if string(got) == string(want) {
			return true
		}
	}
	return false
}

func doclingHybridPageCount(content map[string]any) int {
	switch pages := content["pages"].(type) {
	case map[string]any:
		return len(pages)
	case []any:
		return len(pages)
	default:
		// Standalone images are one logical page even when a converter omits the
		// pages catalog. Text positions still come from item provenance when it
		// exists; this fallback is metadata only.
		return 1
	}
}
