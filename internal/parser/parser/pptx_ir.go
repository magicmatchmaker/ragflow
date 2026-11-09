//
// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package parser

import (
	"encoding/json"
	"fmt"
	"strings"

	"ragflow/internal/entity"
)

const (
	presentationEMUPerTwip       = int64(635)
	defaultPresentationWidthEMU  = int64(12192000)
	defaultPresentationHeightEMU = int64(6858000)
)

const (
	presentationCanvasWidthKey   = "slide_canvas_width_emu"
	presentationCanvasHeightKey  = "slide_canvas_height_emu"
	presentationElementXKey      = "slide_x_emu"
	presentationElementYKey      = "slide_y_emu"
	presentationElementWidthKey  = "slide_element_width_emu"
	presentationElementHeightKey = "slide_element_height_emu"
)

type presentationGeometry struct {
	x, y, width, height int64
}

func (g presentationGeometry) valid() bool { return g.width > 0 && g.height > 0 }

func (g presentationGeometry) apply(item map[string]any) {
	if !g.valid() {
		return
	}
	item[presentationElementXKey] = g.x
	item[presentationElementYKey] = g.y
	item[presentationElementWidthKey] = g.width
	item[presentationElementHeightKey] = g.height
}

func presentationElementGeometry(el docxIRElement, inherited presentationGeometry) presentationGeometry {
	geometry := inherited
	if el.WidthEMU > 0 && el.HeightEMU > 0 {
		geometry = presentationGeometry{x: el.XEMU, y: el.YEMU, width: el.WidthEMU, height: el.HeightEMU}
	}
	if el.Type == "image" && el.DisplayWidthEMU > 0 && el.DisplayHeightEMU > 0 {
		if !geometry.valid() {
			geometry.x, geometry.y = el.XEMU, el.YEMU
		}
		geometry.width, geometry.height = el.DisplayWidthEMU, el.DisplayHeightEMU
	}
	return geometry
}

func presentationCanvas(section docxIRSection) (int64, int64) {
	width := section.PageSetup.WidthTwips * presentationEMUPerTwip
	height := section.PageSetup.HeightTwips * presentationEMUPerTwip
	if width <= 0 || height <= 0 {
		return defaultPresentationWidthEMU, defaultPresentationHeightEMU
	}
	return width, height
}

// buildPPTXJSONSections emits text runs, tables and images in slide order.
// Every item retains its slide number; a blank slide emits an empty text item.
// Container elements are traversed recursively so grouped tables remain structured.
// Image bytes are stored in the run media cache and emitted as media_ref; a
// nil sink is the text-only path and skips images.
func buildPPTXJSONSections(irJSON string, sink *officeMediaSink) ([]map[string]any, error) {
	var ir docxIRDocument
	if err := json.Unmarshal([]byte(irJSON), &ir); err != nil {
		return nil, fmt.Errorf("presentation ir-json decode: %w", err)
	}
	items := make([]map[string]any, 0, len(ir.Sections))
	for i, sec := range ir.Sections {
		start := len(items)
		var lines []string
		var lineGeometry presentationGeometry
		flushText := func() {
			if len(lines) > 0 {
				item := map[string]any{"text": strings.Join(lines, "\n"), "doc_type_kwd": "text"}
				lineGeometry.apply(item)
				items = append(items, item)
				lines = nil
				lineGeometry = presentationGeometry{}
			}
		}
		appendText := func(text string, geometry presentationGeometry) {
			text = strings.TrimSpace(text)
			if text == "" {
				return
			}
			if len(lines) > 0 && geometry != lineGeometry {
				flushText()
			}
			if len(lines) == 0 {
				lineGeometry = geometry
			}
			lines = append(lines, text)
		}
		tableSequence := 0
		var emitElements func([]docxIRElement, presentationGeometry)
		emitElements = func(elements []docxIRElement, inherited presentationGeometry) {
			for _, el := range elements {
				if sink != nil && sink.err != nil {
					return
				}
				geometry := presentationElementGeometry(el, inherited)
				var units []map[string]any
				switch el.Type {
				case "table":
					if len(el.Rows) == 0 {
						continue
					}
					flushText()
					tableSequence++
					before := len(items)
					items = appendOfficeTableSections(items, el, "", fmt.Sprintf("pptx-slide-%d-table-%d", i+1, tableSequence), sink)
					for _, item := range items[before:] {
						geometry.apply(item)
					}
					continue
				case "text_box":
					emitElements(el.contentBlocks(), geometry)
					continue
				case "list":
					for _, item := range el.Items {
						emitElements(item.Content, geometry)
						if item.Nested != nil {
							emitElements([]docxIRElement{{Type: "list", Items: item.Nested.Items}}, geometry)
						}
					}
					continue
				case "paragraph", "heading":
					units = appendDOCXParagraphSections(nil, el, false, sink)
				case "image":
					units = appendDOCXImageSection(nil, el.Data, sink, nil)
				default:
					if text := strings.TrimSpace(docxElementText(el, "\n")); text != "" {
						appendText(text, geometry)
					}
				}
				for _, unit := range units {
					if unit["doc_type_kwd"] == "text" {
						appendText(unit["text"].(string), geometry)
						continue
					}
					flushText()
					geometry.apply(unit)
					items = append(items, unit)
				}
			}
		}
		emitElements(sec.Elements, presentationGeometry{})
		flushText()
		if len(items) == start {
			items = append(items, map[string]any{"text": "", "doc_type_kwd": "text"})
		}
		canvasWidth, canvasHeight := presentationCanvas(sec)
		mediaOrder := 0
		for _, item := range items[start:] {
			item["slide_number"] = i + 1
			item[presentationCanvasWidthKey] = canvasWidth
			item[presentationCanvasHeightKey] = canvasHeight
			if item["doc_type_kwd"] == "image" {
				mediaOrder++
				item["media_order"] = mediaOrder
			}
		}
	}
	return items, nil
}

// itemsAllEmpty reports whether a presentation has no text, tables or images.
func itemsAllEmpty(items []map[string]any) bool {
	for _, it := range items {
		if table, ok := it["table"].(*entity.TableData); ok && table != nil && len(table.Grid) > 0 {
			return false
		}
		if text, _ := it["text"].(string); strings.TrimSpace(text) != "" {
			return false
		}
		if entity.MediaRefFromAny(it["media_ref"]) != nil {
			return false
		}
	}
	return true
}

// itemsFromPlainText wraps whole-document plain text as a single JSON
// item. It salvages a still-readable deck when the structured IR cannot
// be used (IR serialization failure or a sectionless IR); a document
// without extractable text yields an empty, non-nil item slice.
func itemsFromPlainText(text string) []map[string]any {
	items := []map[string]any{}
	if trimmed := strings.TrimSpace(text); trimmed != "" {
		items = append(items, map[string]any{"text": trimmed, "doc_type_kwd": "text"})
	}
	return items
}
