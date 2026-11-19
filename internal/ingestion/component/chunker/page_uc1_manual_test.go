//go:build cgo && manual

package chunker

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"image/png"
	"os"
	"testing"
	"unicode"

	nativeanalyzer "ragflow/internal/deepdoc/parser/pdf/inference/native_analyzer"
	deepdoctype "ragflow/internal/deepdoc/parser/type"
	"ragflow/internal/entity"
	"ragflow/internal/ingestion/component"
	"ragflow/internal/ingestion/component/schema"
	"ragflow/internal/parser/media"
	"ragflow/internal/parser/parser"
	"ragflow/internal/storage"
	"ragflow/internal/utility"
)

const uc1SourceSHA256 = "049be6d8626e7acecf17b5c536797b9f373b7d53ffe6909aa3ac3dbf5663d1d0"
const uc1PDFSourceSHA256 = "be77dc6d2452c3b025b08ecd2ec99d99a0281e10e7dd0bcad9c267c4978cdf15"

// These are SHA-256 digests of three approved, non-secret representative
// search markers after letters-and-digits normalization. Keeping only digests
// in the regression fixture prevents document text from appearing in source or
// failure output.
var uc1SearchMarkerSHA256 = [...]string{
	"91da14f4f9d1088f4e61afb12a245f8a82319d75f09c20333a2988cc92912f7c",
	"fe6acd37dd2c479157cc39847dc48b220e6fa5b2c1bc11b220e8693244213d16",
	"e2e1efb2ea75dbd41cb4d000a83fa7df44e52169b0618298e31bbe9330e1e97e",
}

// Digests of two approved numeric markers visible inside one rasterized PDF
// page. The raw values intentionally stay out of source and test output.
var uc1PDFOCRMarkerSHA256 = [...]string{
	"8d284220975da66c6cba4a32aaedfa0b897187d86ed9f7a2419c9bd10a103505",
	"888df25ae35772424a560c7152a1de794440e0ea5cfee62828333a456a506e05",
}

type presentationManualCache struct {
	data map[string][]byte
}

func containsUC1SearchMarker(text, wantDigest string) bool {
	normalized := make([]rune, 0, len(text))
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			normalized = append(normalized, unicode.ToLower(r))
		}
	}
	for start := range normalized {
		for width := 2; width <= 12 && start+width <= len(normalized); width++ {
			sum := sha256.Sum256([]byte(string(normalized[start : start+width])))
			if hex.EncodeToString(sum[:]) == wantDigest {
				return true
			}
		}
	}
	return false
}

func (c *presentationManualCache) Put(_ context.Context, format, mime string, data []byte) (entity.MediaRef, error) {
	if c.data == nil {
		c.data = make(map[string][]byte)
	}
	sum := sha256.Sum256(data)
	digest := hex.EncodeToString(sum[:])
	c.data[digest] = append([]byte(nil), data...)
	return entity.MediaRef{Format: format, Kind: entity.MediaKindKV, Scope: "uc1-test", SHA256: digest, MIME: mime, Size: int64(len(data))}, nil
}

func (c *presentationManualCache) Get(_ context.Context, ref entity.MediaRef) ([]byte, error) {
	data, ok := c.data[ref.SHA256]
	if !ok {
		return nil, media.ErrMediaNotFound
	}
	return media.ValidatePayload(append([]byte(nil), data...), ref)
}

// TestOchangDoclingPresentationRegression exercises the combined production
// path on an operator-supplied PPTX without printing source text or image
// bytes. Docling must retain semantic chunks, while the native presentation
// path contributes exactly one resolvable full-slide PNG per slide.
func TestOchangDoclingPresentationRegression(t *testing.T) {
	path := os.Getenv("RAGFLOW_OCHANG_PPTX_PATH")
	if path == "" {
		t.Skip("RAGFLOW_OCHANG_PPTX_PATH is not set")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	cache := &presentationManualCache{}
	ctx := media.WithCache(t.Context(), cache)
	parsed := parser.NewDoclingHybridParser(utility.FileTypePPTX).ParseWithResult(ctx, "ochang-regression.pptx", data)
	if parsed.Err != nil {
		t.Fatal(parsed.Err)
	}

	semanticCount := 0
	slides := make(map[int]struct{})
	for _, item := range parser.ItemsToMaps(parsed.JSON) {
		if marked, _ := item[doclingPresentationPreviewSourceKey].(bool); marked {
			if slide, ok := item["slide_number"].(int); ok && slide > 0 {
				slides[slide] = struct{}{}
			}
			continue
		}
		semanticCount++
	}
	if semanticCount == 0 || len(slides) == 0 {
		t.Fatalf("combined parse counts semantic=%d slides=%d", semanticCount, len(slides))
	}

	chunkerComponent, err := NewDoclingHybridChunker(nil)
	if err != nil {
		t.Fatal(err)
	}
	out, err := chunkerComponent.Invoke(ctx, nil, map[string]any{
		"name": "ochang-regression.pptx", "file_type": "pptx", "doc_id": "ochang-regression",
		"output_format": "json", "json": parsed.JSON,
	})
	if err != nil {
		t.Fatal(err)
	}
	chunks, ok := schema.ChunkCarrierToMaps(out["chunks"])
	if !ok {
		t.Fatalf("unexpected chunk carrier %T", out["chunks"])
	}
	previewCount := 0
	for _, chunk := range chunks {
		if preview, _ := chunk[doclingSlidePreviewKey].(bool); !preview {
			continue
		}
		previewCount++
		data, _, err := component.ResolveMediaRef(ctx, chunk)
		if err != nil {
			t.Fatalf("preview %d resolve: %v", previewCount, err)
		}
		if _, err := png.Decode(bytes.NewReader(data)); err != nil {
			t.Fatalf("preview %d PNG: %v", previewCount, err)
		}
	}
	if previewCount != len(slides) {
		t.Fatalf("full-slide previews=%d slides=%d semantic=%d total=%d", previewCount, len(slides), semanticCount, len(chunks))
	}
}

// TestUC1PresentationRegression verifies the real 52-slide source without
// printing document text, filenames, or image data. Enable it explicitly with
// the manual tier and RAGFLOW_UC1_PPTX_PATH.
func TestUC1PresentationRegression(t *testing.T) {
	path := os.Getenv("RAGFLOW_UC1_PPTX_PATH")
	if path == "" {
		t.Skip("RAGFLOW_UC1_PPTX_PATH is not set")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sourceSum := sha256.Sum256(data)
	if got := hex.EncodeToString(sourceSum[:]); got != uc1SourceSHA256 {
		t.Fatalf("source digest = %s, want approved UC1 digest", got)
	}

	cache := &presentationManualCache{}
	ctx := media.WithCache(t.Context(), cache)
	parsed := parser.NewPPTXParser().ParseWithResult(ctx, "uc1-regression.pptx", data)
	if parsed.Err != nil {
		t.Fatal(parsed.Err)
	}
	imageItems := 0
	distinctSourceImages := make(map[string]struct{})
	for _, item := range parsed.JSON {
		m := item.ToMap()
		if m["doc_type_kwd"] != "image" {
			continue
		}
		imageItems++
		if ref := entity.MediaRefFromAny(m["media_ref"]); ref != nil {
			distinctSourceImages[ref.SHA256] = struct{}{}
		}
	}
	if imageItems != 70 || len(distinctSourceImages) != 61 {
		t.Fatalf("source media counts = %d items / %d distinct, want 70 / 61", imageItems, len(distinctSourceImages))
	}

	comp, err := NewPageChunker(nil)
	if err != nil {
		t.Fatal(err)
	}
	out, err := comp.Invoke(ctx, nil, map[string]any{
		"name": "uc1-regression.pptx", "file_type": "pptx", "output_format": "json", "json": parsed.JSON,
	})
	if err != nil {
		t.Fatal(err)
	}
	chunks, ok := schema.ChunkCarrierToMaps(out["chunks"])
	if !ok || len(chunks) != 52 {
		t.Fatalf("slide chunks = %d, want 52", len(chunks))
	}

	previewHash := sha256.New()
	for i, chunk := range chunks {
		if chunk["doc_type_kwd"] != "image" {
			t.Fatalf("slide %d has no preview type", i+1)
		}
		preview, _, err := component.ResolveMediaRef(ctx, chunk)
		if err != nil {
			t.Fatalf("slide %d preview resolve: %v", i+1, err)
		}
		if _, err := png.Decode(bytes.NewReader(preview)); err != nil {
			t.Fatalf("slide %d preview PNG: %v", i+1, err)
		}
		previewHash.Write(preview)
	}
	for markerIndex, markerDigest := range uc1SearchMarkerSHA256 {
		found := false
		for _, chunk := range chunks {
			text, _ := chunk["text"].(string)
			if containsUC1SearchMarker(text, markerDigest) {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("approved search marker %d is absent from all slide chunks", markerIndex+1)
		}
	}
	t.Logf("UC1 regression verified: source_sha256=%s slides=%d image_items=%d distinct_images=%d preview_set_sha256=%s",
		uc1SourceSHA256, len(chunks), imageItems, len(distinctSourceImages), hex.EncodeToString(previewHash.Sum(nil)))
}

// TestUC1PDFPresentationRegression exercises the production PDF parser and
// PageChunker against the approved 52-page conversion. It logs only counts and
// hashes; document text and the source filename never enter test output.
func TestUC1PDFPresentationRegression(t *testing.T) {
	path, modelDir := os.Getenv("RAGFLOW_UC1_PDF_PATH"), os.Getenv("MODEL_DIR")
	if path == "" || modelDir == "" {
		t.Skip("RAGFLOW_UC1_PDF_PATH and MODEL_DIR are required")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sourceSum := sha256.Sum256(data)
	if got := hex.EncodeToString(sourceSum[:]); got != uc1PDFSourceSHA256 {
		t.Fatalf("source digest = %s, want approved UC1 PDF digest", got)
	}

	previousAnalyzer := deepdoctype.NativeDocAnalyzerFactory
	t.Cleanup(func() { deepdoctype.SetNativeDocAnalyzerFactory(previousAnalyzer) })
	if err := nativeanalyzer.Register(modelDir, nativeanalyzer.DefaultDropScore); err != nil {
		t.Fatal(err)
	}
	parsed := parser.NewPDFParser().ParseWithResult(t.Context(), "uc1-regression.pdf", data)
	if parsed.Err != nil {
		t.Fatal(parsed.Err)
	}

	store := storage.NewMemoryStorage()
	if err := store.Put(t.Context(), "uc1-pdf-test", "source.pdf", data); err != nil {
		t.Fatal(err)
	}
	previousStorage := component.SetStorageFactoryOverride
	component.SetStorageFactoryOverride = func() storage.Storage { return store }
	t.Cleanup(func() { component.SetStorageFactoryOverride = previousStorage })

	cache := &presentationManualCache{}
	ctx := media.WithCache(t.Context(), cache)
	pageChunker, err := NewPageChunker(nil)
	if err != nil {
		t.Fatal(err)
	}
	out, err := pageChunker.Invoke(ctx, nil, map[string]any{
		"name":          "uc1-regression.pdf",
		"file_type":     "pdf",
		"output_format": "json",
		"json":          parsed.JSON,
		"file":          parsed.File,
		"bucket":        "uc1-pdf-test",
		"path":          "source.pdf",
	})
	if err != nil {
		t.Fatal(err)
	}
	chunks, ok := schema.ChunkCarrierToMaps(out["chunks"])
	if !ok || len(chunks) != 52 {
		t.Fatalf("PDF page chunks = %d, want 52", len(chunks))
	}

	previewHash := sha256.New()
	for pageIndex, chunk := range chunks {
		for _, key := range []string{"positions", "_pdf_positions"} {
			encoded, err := json.Marshal(chunk[key])
			if err != nil {
				t.Fatalf("page %d %s encode: %v", pageIndex+1, key, err)
			}
			var rows [][]float64
			if err := json.Unmarshal(encoded, &rows); err != nil || len(rows) == 0 {
				t.Fatalf("page %d %s missing or invalid", pageIndex+1, key)
			}
			for _, row := range rows {
				if len(row) != 5 || row[0] != float64(pageIndex) {
					t.Fatalf("page %d %s has noncanonical page index", pageIndex+1, key)
				}
			}
		}
		preview, _, err := component.ResolveMediaRef(ctx, chunk)
		if err != nil {
			t.Fatalf("page %d preview resolve: %v", pageIndex+1, err)
		}
		if _, err := png.Decode(bytes.NewReader(preview)); err != nil {
			t.Fatalf("page %d preview PNG: %v", pageIndex+1, err)
		}
		previewHash.Write(preview)
	}

	const approvedOCRPageIndex = 28
	pageText, _ := chunks[approvedOCRPageIndex]["text"].(string)
	for markerIndex, markerDigest := range uc1PDFOCRMarkerSHA256 {
		if !containsUC1SearchMarker(pageText, markerDigest) {
			t.Fatalf("approved OCR marker %d is absent from its PDF page chunk", markerIndex+1)
		}
	}
	t.Logf("UC1 PDF regression verified: source_sha256=%s pages=%d previews=%d preview_set_sha256=%s",
		uc1PDFSourceSHA256, len(chunks), len(chunks), hex.EncodeToString(previewHash.Sum(nil)))
}
