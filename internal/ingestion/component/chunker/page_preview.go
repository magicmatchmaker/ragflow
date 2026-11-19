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
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"html"
	"image"
	"image/color"
	stddraw "image/draw"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"math"
	"os"
	"strings"
	"sync"
	"unicode"

	xdraw "golang.org/x/image/draw"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"

	imageparser "ragflow/internal/deepdoc/parser/image"
	deepdoctype "ragflow/internal/deepdoc/parser/type"
	"ragflow/internal/ingestion/component"
	"ragflow/internal/ingestion/component/schema"
	"ragflow/internal/parser/media"
	parserlib "ragflow/internal/parser/parser"
	"ragflow/internal/parser/tableutil"
)

const (
	presentationCanvasWidthKey   = "slide_canvas_width_emu"
	presentationCanvasHeightKey  = "slide_canvas_height_emu"
	presentationElementXKey      = "slide_x_emu"
	presentationElementYKey      = "slide_y_emu"
	presentationElementWidthKey  = "slide_element_width_emu"
	presentationElementHeightKey = "slide_element_height_emu"

	defaultPresentationWidthEMU  = int64(12192000)
	defaultPresentationHeightEMU = int64(6858000)

	maxPresentationPreviewWidth  = 1600
	maxPresentationPreviewHeight = 1200
	maxPresentationPreviewPixels = 2_000_000
	maxPresentationSourcePixels  = 64_000_000
	maxPresentationTextRunes     = 4000
	maxPresentationOCRRunes      = 8000
)

var (
	presentationFontOnce sync.Once
	// opentype.Face keeps mutable glyph-loading buffers and is not safe for
	// concurrent use. Presentation ingestion runs several documents in
	// parallel, so sharing the face without serialization can corrupt those
	// buffers and panic inside sfnt.LoadGlyph.
	presentationFontMu   sync.Mutex
	presentationFontFace font.Face = basicfont.Face7x13
)

// composePresentationPreview renders all slide images and positioned text in
// parser order. That order is the PPTX shape order and therefore also the
// preview z-order. Source images are decoded under strict byte/pixel limits;
// coordinates are projected onto a bounded canvas and clipped by draw.Draw.
func composePresentationPreview(ctx context.Context, items []schema.ChunkDoc) ([]byte, int, error) {
	canvasWidthEMU, canvasHeightEMU := presentationCanvasDimensions(items)
	width, height := boundedPresentationCanvas(canvasWidthEMU, canvasHeightEMU)
	canvas := image.NewRGBA(image.Rect(0, 0, width, height))
	stddraw.Draw(canvas, canvas.Bounds(), image.NewUniform(color.White), image.Point{}, stddraw.Src)

	mediaCount := 0
	for _, item := range items {
		if item.MediaRef != nil || item.Image != "" {
			mediaCount++
		}
	}
	var composeErrors []error
	mediaIndex, textIndex, renderedMedia := 0, 0, 0
	for _, item := range items {
		if err := ctx.Err(); err != nil {
			return nil, mediaCount, err
		}
		if item.MediaRef != nil || item.Image != "" {
			data, err := presentationImageBytes(ctx, item)
			if err != nil {
				composeErrors = append(composeErrors, err)
				mediaIndex++
				continue
			}
			img, err := decodeBoundedPresentationImage(data)
			if err != nil {
				composeErrors = append(composeErrors, err)
				mediaIndex++
				continue
			}
			dst, ok := presentationElementRect(item, canvasWidthEMU, canvasHeightEMU, width, height)
			if !ok {
				dst = fallbackPresentationMediaRect(mediaIndex, mediaCount, width, height)
			}
			dst = fitImageRect(dst, img.Bounds().Dx(), img.Bounds().Dy())
			if !dst.Empty() {
				xdraw.CatmullRom.Scale(canvas, dst, img, img.Bounds(), stddraw.Over, nil)
				renderedMedia++
			}
			mediaIndex++
		}

		text := stripPresentationMarkup(item.Text)
		if text == "" {
			continue
		}
		dst, ok := presentationElementRect(item, canvasWidthEMU, canvasHeightEMU, width, height)
		if !ok {
			dst = fallbackPresentationTextRect(textIndex, width, height)
			textIndex++
		}
		if err := drawPresentationText(canvas, dst, text); err != nil {
			composeErrors = append(composeErrors, err)
		}
	}
	if mediaCount > 0 && renderedMedia == 0 && len(composeErrors) == 0 {
		composeErrors = append(composeErrors, errors.New("presentation media produced no visible preview"))
	}

	var encoded bytes.Buffer
	if err := png.Encode(&encoded, canvas); err != nil {
		return nil, mediaCount, err
	}
	if encoded.Len() > media.ImageMaxBytes() {
		return nil, mediaCount, errors.New("presentation preview exceeds image byte limit")
	}
	return encoded.Bytes(), mediaCount, errors.Join(composeErrors...)
}

func presentationImageBytes(ctx context.Context, item schema.ChunkDoc) ([]byte, error) {
	if item.MediaRef != nil {
		data, _, err := component.ResolveMediaRef(ctx, item.ToMap())
		return data, err
	}
	return decodeChunkImage(item.Image)
}

func decodeBoundedPresentationImage(data []byte) (image.Image, error) {
	if len(data) == 0 || len(data) > media.ImageMaxBytes() {
		return nil, errors.New("presentation image byte limit exceeded")
	}
	config, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	if config.Width <= 0 || config.Height <= 0 || config.Width > maxPresentationSourcePixels/config.Height {
		return nil, errors.New("presentation image dimensions are invalid")
	}
	if config.Width*config.Height > maxPresentationSourcePixels {
		return nil, errors.New("presentation image pixel limit exceeded")
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	return img, err
}

func presentationCanvasDimensions(items []schema.ChunkDoc) (int64, int64) {
	for _, item := range items {
		width, widthOK := presentationExtraInt64(item, presentationCanvasWidthKey)
		height, heightOK := presentationExtraInt64(item, presentationCanvasHeightKey)
		if widthOK && heightOK && width > 0 && height > 0 {
			return width, height
		}
	}
	return defaultPresentationWidthEMU, defaultPresentationHeightEMU
}

func boundedPresentationCanvas(widthEMU, heightEMU int64) (int, int) {
	if widthEMU <= 0 || heightEMU <= 0 {
		widthEMU, heightEMU = defaultPresentationWidthEMU, defaultPresentationHeightEMU
	}
	width, height := float64(widthEMU), float64(heightEMU)
	scale := math.Min(float64(maxPresentationPreviewWidth)/width, float64(maxPresentationPreviewHeight)/height)
	if pixels := width * height * scale * scale; pixels > maxPresentationPreviewPixels {
		scale *= math.Sqrt(maxPresentationPreviewPixels / pixels)
	}
	outWidth := max(1, int(math.Round(width*scale)))
	outHeight := max(1, int(math.Round(height*scale)))
	return outWidth, outHeight
}

func presentationElementRect(item schema.ChunkDoc, canvasWidthEMU, canvasHeightEMU int64, width, height int) (image.Rectangle, bool) {
	x, xOK := presentationExtraInt64(item, presentationElementXKey)
	y, yOK := presentationExtraInt64(item, presentationElementYKey)
	elementWidth, widthOK := presentationExtraInt64(item, presentationElementWidthKey)
	elementHeight, heightOK := presentationExtraInt64(item, presentationElementHeightKey)
	if !xOK || !yOK || !widthOK || !heightOK || elementWidth <= 0 || elementHeight <= 0 || canvasWidthEMU <= 0 || canvasHeightEMU <= 0 {
		return image.Rectangle{}, false
	}
	x0 := clampFloat(float64(x)/float64(canvasWidthEMU)*float64(width), 0, float64(width))
	y0 := clampFloat(float64(y)/float64(canvasHeightEMU)*float64(height), 0, float64(height))
	x1 := clampFloat(float64(x+elementWidth)/float64(canvasWidthEMU)*float64(width), 0, float64(width))
	y1 := clampFloat(float64(y+elementHeight)/float64(canvasHeightEMU)*float64(height), 0, float64(height))
	rect := image.Rect(int(math.Round(x0)), int(math.Round(y0)), int(math.Round(x1)), int(math.Round(y1))).Intersect(image.Rect(0, 0, width, height))
	return rect, !rect.Empty()
}

func presentationExtraInt64(item schema.ChunkDoc, key string) (int64, bool) {
	raw, ok := item.Extra[key]
	if !ok || len(raw) == 0 {
		return 0, false
	}
	var value int64
	if err := json.Unmarshal(raw, &value); err != nil {
		return 0, false
	}
	return value, true
}

func clampFloat(value, low, high float64) float64 {
	return math.Min(high, math.Max(low, value))
}

func fallbackPresentationMediaRect(index, count, width, height int) image.Rectangle {
	columns := max(1, int(math.Ceil(math.Sqrt(float64(count)))))
	rows := max(1, (count+columns-1)/columns)
	column, row := index%columns, index/columns
	return image.Rect(column*width/columns, row*height/rows, (column+1)*width/columns, (row+1)*height/rows)
}

func fallbackPresentationTextRect(index, width, height int) image.Rectangle {
	bandHeight := max(48, height/6)
	top := min(height-bandHeight, index*bandHeight)
	return image.Rect(8, max(0, top), max(9, width-8), min(height, top+bandHeight))
}

func fitImageRect(dst image.Rectangle, sourceWidth, sourceHeight int) image.Rectangle {
	if dst.Empty() || sourceWidth <= 0 || sourceHeight <= 0 {
		return image.Rectangle{}
	}
	scale := math.Min(float64(dst.Dx())/float64(sourceWidth), float64(dst.Dy())/float64(sourceHeight))
	width := max(1, int(math.Round(float64(sourceWidth)*scale)))
	height := max(1, int(math.Round(float64(sourceHeight)*scale)))
	x := dst.Min.X + (dst.Dx()-width)/2
	y := dst.Min.Y + (dst.Dy()-height)/2
	return image.Rect(x, y, x+width, y+height)
}

func encodeBoundedPresentationImage(source image.Image) ([]byte, error) {
	if source == nil || source.Bounds().Dx() <= 0 || source.Bounds().Dy() <= 0 {
		return nil, errors.New("presentation preview source is empty")
	}
	width, height := source.Bounds().Dx(), source.Bounds().Dy()
	if width > maxPresentationSourcePixels/height || width*height > maxPresentationSourcePixels {
		return nil, errors.New("presentation preview source exceeds pixel limit")
	}
	targetWidth, targetHeight := boundedPresentationCanvas(int64(width), int64(height))
	target := image.NewRGBA(image.Rect(0, 0, targetWidth, targetHeight))
	stddraw.Draw(target, target.Bounds(), image.NewUniform(color.White), image.Point{}, stddraw.Src)
	xdraw.CatmullRom.Scale(target, target.Bounds(), source, source.Bounds(), stddraw.Over, nil)
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, target); err != nil {
		return nil, err
	}
	if encoded.Len() > media.ImageMaxBytes() {
		return nil, errors.New("presentation preview exceeds image byte limit")
	}
	return encoded.Bytes(), nil
}

func drawPresentationText(dst *image.RGBA, rect image.Rectangle, text string) (err error) {
	// Keep a malformed font/glyph from aborting the entire document. The slide
	// preview remains useful with its other text and images, and the caller logs
	// this as a partial rendering error.
	defer func() {
		if recovered := recover(); recovered != nil {
			err = errors.New("presentation text renderer panicked")
		}
	}()
	presentationFontMu.Lock()
	defer presentationFontMu.Unlock()

	rect = rect.Intersect(dst.Bounds())
	if rect.Dx() < 8 || rect.Dy() < 8 || text == "" {
		return nil
	}
	face := getPresentationFontFace()
	lineHeight := max(1, face.Metrics().Height.Ceil())
	maxLines := max(1, (rect.Dy()-8)/lineHeight)
	lines := wrapPresentationText(text, face, max(1, rect.Dx()-8), maxLines)
	if len(lines) == 0 {
		return nil
	}
	// A translucent white backing keeps text readable over screenshots while
	// preserving the image underneath and its alpha compositing.
	backing := image.NewUniform(color.RGBA{R: 255, G: 255, B: 255, A: 220})
	stddraw.Draw(dst, rect, backing, image.Point{}, stddraw.Over)
	drawer := font.Drawer{Dst: dst, Src: image.NewUniform(color.Black), Face: face}
	y := rect.Min.Y + 4 + face.Metrics().Ascent.Ceil()
	for _, line := range lines {
		drawer.Dot = fixedPoint(rect.Min.X+4, y)
		drawer.DrawString(line)
		y += lineHeight
	}
	return nil
}

func fixedPoint(x, y int) fixed.Point26_6 {
	return fixed.P(x, y)
}

func wrapPresentationText(text string, face font.Face, maxWidth, maxLines int) []string {
	runes := []rune(text)
	if len(runes) > maxPresentationTextRunes {
		runes = runes[:maxPresentationTextRunes]
	}
	var lines []string
	var line []rune
	flush := func() {
		if len(line) > 0 && len(lines) < maxLines {
			lines = append(lines, strings.TrimSpace(string(line)))
			line = nil
		}
	}
	for _, r := range runes {
		if len(lines) >= maxLines {
			break
		}
		if r == '\n' {
			flush()
			continue
		}
		candidate := append(append([]rune(nil), line...), r)
		if len(line) > 0 && font.MeasureString(face, string(candidate)).Ceil() > maxWidth {
			flush()
		}
		line = append(line, r)
	}
	flush()
	return lines
}

func stripPresentationMarkup(text string) string {
	text = html.UnescapeString(text)
	var out strings.Builder
	inTag := false
	for _, r := range text {
		switch r {
		case '<':
			inTag = true
		case '>':
			if inTag {
				inTag = false
				out.WriteByte(' ')
			}
		default:
			if !inTag {
				out.WriteRune(r)
			}
		}
	}
	return strings.TrimSpace(out.String())
}

func getPresentationFontFace() font.Face {
	presentationFontOnce.Do(func() {
		for _, path := range []string{
			"/usr/local/share/fonts/truetype/noto/NotoSansCJKsc-VF.ttf",
			"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
		} {
			data, err := os.ReadFile(path)
			if err != nil {
				continue
			}
			parsed, err := opentype.Parse(data)
			if err != nil {
				continue
			}
			face, err := opentype.NewFace(parsed, &opentype.FaceOptions{Size: 16, DPI: 72, Hinting: font.HintingFull})
			if err == nil {
				presentationFontFace = face
				return
			}
		}
	})
	return presentationFontFace
}

func storePresentationPreview(ctx context.Context, data []byte) (*schema.ChunkDoc, error) {
	if len(data) == 0 {
		return nil, nil
	}
	doc := &schema.ChunkDoc{DocType: "image"}
	if cache := media.CacheFromContext(ctx); cache != nil {
		ref, err := cache.Put(ctx, "presentation-preview", "image/png", data)
		if err != nil {
			return nil, err
		}
		doc.MediaRef = &ref
		return doc, nil
	}
	doc.Image = "data:image/png;base64," + base64.StdEncoding.EncodeToString(data)
	return doc, nil
}

func clearPresentationPreviewMetadata(item *schema.ChunkDoc) {
	if item == nil || item.Extra == nil {
		return
	}
	for _, key := range []string{
		presentationCanvasWidthKey,
		presentationCanvasHeightKey,
		presentationElementXKey,
		presentationElementYKey,
		presentationElementWidthKey,
		presentationElementHeightKey,
		"media_order",
		"source_table_id",
		"parent_table_id",
	} {
		delete(item.Extra, key)
	}
}

func resolvePresentationOCRAnalyzer() (analyzer deepdoctype.DocAnalyzer, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			analyzer = nil
			err = errors.New("local OCR analyzer initialization panicked")
		}
	}()
	factory := deepdoctype.NativeDocAnalyzerFactory
	if factory == nil {
		return nil, errors.New("local OCR analyzer is not registered")
	}
	analyzer, ok := factory()
	if !ok || analyzer == nil || !analyzer.Health() {
		return nil, errors.New("local OCR analyzer is unavailable")
	}
	return analyzer, nil
}

func extractPresentationOCR(ctx context.Context, preview []byte, analyzer deepdoctype.DocAnalyzer) (text string, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			text = ""
			err = errors.New("local OCR analyzer panicked")
		}
	}()
	if analyzer == nil {
		return "", errors.New("local OCR analyzer is unavailable")
	}
	img, err := png.Decode(bytes.NewReader(preview))
	if err != nil {
		return "", err
	}
	release, err := parserlib.AcquireImageMedia(ctx)
	if err != nil {
		return "", err
	}
	defer release()
	blocks, err := imageparser.Parse(ctx, img, analyzer)
	if err != nil {
		return "", err
	}
	parts := make([]string, 0, len(blocks))
	for _, block := range blocks {
		blockText := strings.TrimSpace(block.Text)
		if block.Table != nil {
			blockText = strings.TrimSpace(tableutil.RenderTableHTML(block.Table))
		}
		if blockText != "" {
			parts = append(parts, blockText)
		}
	}
	return strings.Join(parts, "\n"), nil
}

// mergePresentationOCR appends only OCR lines not already present in the
// parser text. Matching ignores case and whitespace, which removes the common
// duplicate where PowerPoint text is both extracted from XML and visible in a
// screenshot. The OCR contribution is rune-capped so a pathological slide
// cannot expand a single embedding request without bound.
func mergePresentationOCR(existing, ocr string) string {
	existing = strings.TrimSpace(existing)
	ocr = strings.TrimSpace(ocr)
	if ocr == "" {
		return existing
	}
	seen := make(map[string]struct{})
	existingNormalized := normalizePresentationOCR(existing)
	for _, line := range strings.Split(existing, "\n") {
		if key := normalizePresentationOCR(line); key != "" {
			seen[key] = struct{}{}
		}
	}
	var added []string
	remaining := maxPresentationOCRRunes
	for _, rawLine := range strings.Split(ocr, "\n") {
		line := strings.TrimSpace(rawLine)
		key := normalizePresentationOCR(line)
		if key == "" {
			continue
		}
		if _, duplicate := seen[key]; duplicate || (len([]rune(key)) >= 4 && strings.Contains(existingNormalized, key)) {
			continue
		}
		seen[key] = struct{}{}
		runes := []rune(line)
		if len(runes) > remaining {
			runes = runes[:remaining]
		}
		if len(runes) > 0 {
			added = append(added, string(runes))
			remaining -= len(runes)
		}
		if remaining == 0 {
			break
		}
	}
	if len(added) == 0 {
		return existing
	}
	if existing == "" {
		return strings.Join(added, "\n")
	}
	return existing + "\n\n" + strings.Join(added, "\n")
}

func normalizePresentationOCR(text string) string {
	var normalized strings.Builder
	for _, r := range strings.ToLower(text) {
		if !unicode.IsSpace(r) {
			normalized.WriteRune(r)
		}
	}
	return normalized.String()
}
