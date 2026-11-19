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
	"image"
	"testing"
)

func TestDrawPresentationTextContainsRendererPanics(t *testing.T) {
	if err := drawPresentationText(nil, image.Rect(0, 0, 100, 100), "text"); err == nil {
		t.Fatal("expected a contained renderer panic to be returned as an error")
	}
}

func TestDrawPresentationTextSupportsConcurrentSlides(t *testing.T) {
	const workers = 24
	errs := make(chan error, workers)
	for range workers {
		go func() {
			canvas := image.NewRGBA(image.Rect(0, 0, 640, 360))
			errs <- drawPresentationText(canvas, canvas.Bounds(), "오창 설비 점검 · Wet Coater Inspector")
		}()
	}
	for range workers {
		if err := <-errs; err != nil {
			t.Fatalf("concurrent presentation text rendering failed: %v", err)
		}
	}
}
