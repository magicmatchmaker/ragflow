//go:build cgo && manual

package chunker

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"image/png"
	"os"
	"testing"
	"unicode"

	"ragflow/internal/entity"
	"ragflow/internal/ingestion/component"
	"ragflow/internal/ingestion/component/schema"
	"ragflow/internal/parser/media"
	"ragflow/internal/parser/parser"
)

const uc1SourceSHA256 = "049be6d8626e7acecf17b5c536797b9f373b7d53ffe6909aa3ac3dbf5663d1d0"

// These are SHA-256 digests of three approved, non-secret representative
// search markers after letters-and-digits normalization. Keeping only digests
// in the regression fixture prevents document text from appearing in source or
// failure output.
var uc1SearchMarkerSHA256 = [...]string{
	"91da14f4f9d1088f4e61afb12a245f8a82319d75f09c20333a2988cc92912f7c",
	"fe6acd37dd2c479157cc39847dc48b220e6fa5b2c1bc11b220e8693244213d16",
	"e2e1efb2ea75dbd41cb4d000a83fa7df44e52169b0618298e31bbe9330e1e97e",
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
