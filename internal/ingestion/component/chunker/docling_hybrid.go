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
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"ragflow/internal/agent/runtime"
	"ragflow/internal/common"
	"ragflow/internal/ingestion/component/schema"

	"gorm.io/gorm"
)

// DoclingHybridChunker is intentionally a pass-through. Docling's official
// HybridChunker has already chosen semantic/token-aware boundaries, so running
// PageChunker (render + OCR) or another splitting algorithm would destroy
// those boundaries. Native parser JSON from formats outside Docling's verified
// allowlist is also accepted so a mixed dataset can use one downstream
// workflow without silently dropping its fallback documents.
type DoclingHybridChunker struct{}

func NewDoclingHybridChunker(_ map[string]any) (runtime.Component, error) {
	return &DoclingHybridChunker{}, nil
}

func (c *DoclingHybridChunker) Inputs() map[string]string  { return ChunkerInputs }
func (c *DoclingHybridChunker) Outputs() map[string]string { return ChunkerOutputs }

func (c *DoclingHybridChunker) Invoke(ctx context.Context, _ *gorm.DB, inputs map[string]any) (map[string]any, error) {
	if inputs == nil {
		return inputErrorOutput(errors.New("DoclingHybridChunker received no input")), nil
	}
	upstream, err := decodeChunkerFromUpstream(inputs)
	if err != nil {
		return inputErrorOutput(err), nil
	}
	var chunks []schema.ChunkDoc
	switch upstream.OutputFormat {
	case schema.PayloadFormatChunks:
		chunks = upstream.Chunks
	case schema.PayloadFormatJSON, "":
		chunks = upstream.JSONResult
	case schema.PayloadFormatMarkdown:
		if upstream.MarkdownResult != nil && strings.TrimSpace(*upstream.MarkdownResult) != "" {
			chunks = []schema.ChunkDoc{{Text: *upstream.MarkdownResult, DocType: "text"}}
		}
	case schema.PayloadFormatText:
		if upstream.TextResult != nil && strings.TrimSpace(*upstream.TextResult) != "" {
			chunks = []schema.ChunkDoc{{Text: *upstream.TextResult, DocType: "text"}}
		}
	case schema.PayloadFormatHTML:
		if upstream.HTMLResult != nil && strings.TrimSpace(*upstream.HTMLResult) != "" {
			chunks = []schema.ChunkDoc{{Text: *upstream.HTMLResult, DocType: "text"}}
		}
	}
	if len(chunks) == 0 {
		return inputErrorOutput(errors.New("DoclingHybridChunker received zero chunks; refusing a silent successful ingestion")), nil
	}
	_, docID := resolveImageUploadContext(ctx, inputs)
	chunks, err = expandDoclingPictureMedia(chunks, docID)
	if err != nil {
		return inputErrorOutput(err), nil
	}
	return chunkOutputs(chunks), nil
}

type doclingPictureMedia struct {
	Ref          string          `json:"ref"`
	Image        string          `json:"image"`
	Positions    json.RawMessage `json:"positions"`
	Descriptions []string        `json:"descriptions"`
}

// expandDoclingPictureMedia bridges Docling's many-doc_items-per-hybrid-chunk
// model to RAGFlow's one-img_id-per-chunk index schema. The first media chunk
// keeps the official HybridChunker text while aggregate source provenance is
// retained separately;
// subsequent pictures become linked media chunks whose text is their native
// picture description. Every output owns one image payload, so the existing
// upload decorator creates one durable img_id per picture and removes bytes
// before indexing.
func expandDoclingPictureMedia(chunks []schema.ChunkDoc, docID string) ([]schema.ChunkDoc, error) {
	out := make([]schema.ChunkDoc, 0, len(chunks))
	for _, source := range chunks {
		raw, exists := source.Extra["docling_picture_media"]
		if !exists {
			out = append(out, source)
			continue
		}
		var media []doclingPictureMedia
		if err := json.Unmarshal(raw, &media); err != nil {
			return nil, fmt.Errorf("DoclingHybridChunker decode picture media: %w", err)
		}
		if len(media) == 0 {
			return nil, errors.New("DoclingHybridChunker received empty docling_picture_media")
		}
		if len(media) > 1 {
			parent := source
			parent.Extra = cloneChunkExtra(source.Extra)
			delete(parent.Extra, "docling_picture_media")
			parent.Image = ""
			parent.DocType = "text"
			out = append(out, parent)
		}
		for index, picture := range media {
			if strings.TrimSpace(picture.Image) == "" {
				return nil, fmt.Errorf("DoclingHybridChunker picture %d has no image payload", index)
			}
			chunk := source
			chunk.Extra = cloneChunkExtra(source.Extra)
			delete(chunk.Extra, "docling_picture_media")
			delete(chunk.Extra, "docling_picture_refs")
			delete(chunk.Extra, "docling_picture_descriptions")
			chunk.Image = picture.Image
			chunk.DocType = "image"
			if encoded, err := json.Marshal(picture.Ref); err == nil {
				chunk.Extra["docling_picture_ref"] = encoded
			}
			if encoded, err := json.Marshal(index); err == nil {
				chunk.Extra["docling_media_ordinal"] = encoded
			}
			if encoded, err := json.Marshal(picture.Descriptions); err == nil {
				chunk.Extra["docling_picture_descriptions"] = encoded
			}
			if parent, ok := chunk.Extra["docling_chunk_index"]; ok {
				chunk.Extra["docling_parent_chunk_index"] = append(json.RawMessage(nil), parent...)
			}
			if len(source.Positions) > 0 {
				chunk.Extra["docling_source_positions"] = append(json.RawMessage(nil), source.Positions...)
			}
			if len(picture.Positions) > 0 && string(picture.Positions) != "null" {
				chunk.Positions = append(json.RawMessage(nil), picture.Positions...)
				chunk.PDFPositions = append(json.RawMessage(nil), picture.Positions...)
			}
			if len(media) > 1 {
				description := strings.TrimSpace(strings.Join(picture.Descriptions, "\n"))
				if description == "" {
					description = fmt.Sprintf("[Picture %d/%d]", index+1, len(media))
				}
				chunk.Text = description
			}
			identity, identityErr := doclingPictureIdentity(chunk.Text, picture, index)
			if identityErr != nil {
				return nil, fmt.Errorf("DoclingHybridChunker picture %d identity: %w", index, identityErr)
			}
			chunk.ID = common.ChunkID(docID, identity)
			out = append(out, chunk)
		}
	}
	if len(out) == 0 {
		return nil, errors.New("DoclingHybridChunker picture expansion produced zero chunks")
	}
	return out, nil
}

// doclingPictureIdentity makes every Docling picture a distinct durable media
// object even when two figures have the same description and no geometry.
// Ref and ordinal preserve source identity; the decoded payload digest keeps
// the identity content-addressed across equivalent data-URL wrappers. The
// body remains part of the digest so changed descriptions invalidate caches.
func doclingPictureIdentity(text string, picture doclingPictureMedia, ordinal int) (string, error) {
	payload, err := decodeChunkImage(picture.Image)
	if err != nil {
		return "", fmt.Errorf("decode image payload: %w", err)
	}
	payloadDigest := sha256.Sum256(payload)
	canonical, err := json.Marshal(struct {
		Text          string `json:"text"`
		Ref           string `json:"ref"`
		Ordinal       int    `json:"ordinal"`
		PayloadSHA256 string `json:"payload_sha256"`
	}{
		Text:          text,
		Ref:           strings.TrimSpace(picture.Ref),
		Ordinal:       ordinal,
		PayloadSHA256: hex.EncodeToString(payloadDigest[:]),
	})
	if err != nil {
		return "", err
	}
	identityDigest := sha256.Sum256(canonical)
	return mediaChunkIDPrefix + "docling:" + hex.EncodeToString(identityDigest[:]), nil
}

func cloneChunkExtra(source map[string]json.RawMessage) map[string]json.RawMessage {
	cloned := make(map[string]json.RawMessage, len(source))
	for key, raw := range source {
		cloned[key] = append(json.RawMessage(nil), raw...)
	}
	return cloned
}

func init() {
	MustRegisterChunker(ComponentNameDoclingHybridChunker)
}
