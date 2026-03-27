# Task Board

## Current Sprint

### T1: Core Model — Forward Pass ✅
**Repo:** tiny_waves_transformer (root)
**File:** `src/model.js` (testable standalone), later merged into `WaveTransformer.jsx`
- Xavier-initialized weight matrices per spec
- GELU activation
- Softmax with numerical stability (subtract max)
- Full forward pass: embedding → attention → FFN → logits → probs
- **AC:** `forward([0,1,2,3])` returns valid 16-way probability distribution

### T2: Training — Numerical + Analytical Gradients
**Repo:** tiny_waves_transformer (root)
- Sine wave data generation per spec
- Cross-entropy loss
- Numerical gradient computation (for verification)
- Analytical backward pass (for speed)
- Gradient verification: analytical vs numerical, relative error < 1e-3
- Gradient clipping ±5.0
- **AC:** Loss decreases from ~2.77 to below 1.5 within a few hundred epochs

### T3: React UI Component
**Repo:** tiny_waves_transformer (root)
**File:** `WaveTransformer.jsx`
- Drawing canvas with freehand input and quantization
- Toolbar: Train, Play/Pause, Step ◀▶, Rewind, Clear, Reset
- Status bar: Trail toggle, seed slider, epoch/loss/gen count
- Wave canvas: seed (white) + generated (green) display
- Texture panel: 9 static weight heatmaps + 2 dynamic
- Probability bar during generation
- State machine: DRAW → IDLE → PLAYING ↔ PAUSED
- **AC:** All controls work, phone layout (390px), textures update live

### T4: Integration & Polish
- Merge model + training + UI into single JSX
- End-to-end test: train → draw → generate → verify wave continuation
- Performance check: training tick < 50ms with analytical gradients
- Final cleanup
