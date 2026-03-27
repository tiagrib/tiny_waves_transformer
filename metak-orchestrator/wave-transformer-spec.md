# Wave Transformer — Implementation Spec

## Overview

A tiny autoregressive transformer that generates visual waveforms. Architecturally identical to GPT, but its vocabulary is 16 amplitude levels instead of text tokens. The user draws a wave seed, trains the model on quantized sine waves, then watches it continue their drawing — all while the model's weight matrices are rendered as live heatmap textures.

The deliverable is a single React `.jsx` artifact (no build tools, runs in Claude.ai's artifact renderer or any React sandbox).

---

## Model Architecture

### Hyperparameters

| Name | Value | Notes |
|------|-------|-------|
| `NUM_TOKENS` | 16 | Amplitude levels 0–15 |
| `EMBED_DIM` | 12 | Dimensionality of all vectors |
| `SEQ_LEN` | 24 | Max context window |
| `FFN_DIM` | 36 | 3× embed_dim |
| `NUM_HEADS` | 1 | Single attention head |
| `NUM_LAYERS` | 1 | Single transformer layer |

### Parameters (total: ~2,160)

| Matrix | Shape | Params | Purpose |
|--------|-------|--------|---------|
| `tok_emb` | 16×12 | 192 | Token embedding lookup |
| `pos_emb` | 24×12 | 288 | Positional embedding lookup |
| `Wq` | 12×12 | 144 | Query projection |
| `Wk` | 12×12 | 144 | Key projection |
| `Wv` | 12×12 | 144 | Value projection |
| `Wo` | 12×12 | 144 | Attention output projection |
| `ffn_up` | 36×12 | 432 | FFN expansion (12 → 36) |
| `ffn_up_bias` | 36 | 36 | FFN expansion bias |
| `ffn_down` | 12×36 | 432 | FFN compression (36 → 12) |
| `ffn_down_bias` | 12 | 12 | FFN compression bias |
| `out_proj` | 16×12 | 192 | Readout head (logits) |

### Forward Pass

**No layer norms.** At this scale they add complexity without benefit, and the backward pass through them is error-prone.

```
input tokens: [t0, t1, ..., tN]
     ↓
embedding = tok_emb[ti] + pos_emb[i]         # for each position i
     ↓
Q = Wq × embedding       (per position)
K = Wk × embedding       (per position)
V = Wv × embedding       (per position)
     ↓
scores[i][j] = (Q[i] · K[j]) / √12          # j ≤ i (causal mask)
attn_weights = softmax(scores)                # per row
context = Σ_j  attn_weights[j] × V[j]
attn_out = Wo × context
     ↓
residual_1 = embedding + attn_out
     ↓
ffn_pre = ffn_up × residual_1 + ffn_up_bias
ffn_act = GELU(ffn_pre)
ffn_out = ffn_down × ffn_act + ffn_down_bias
     ↓
residual_2 = residual_1 + ffn_out
     ↓
logits = out_proj × residual_2[last_token]    # only last token
probs = softmax(logits)                       # 16-way distribution
```

### GELU activation

```
gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

### Initialization

All weight matrices: Xavier init, `scale = √(2 / (fan_in + fan_out))`.
Positional embeddings: smaller scale (0.1).
All biases: zeros.

---

## Training

### Data Generation

Generate long quantized sine waves for random window sampling:

```
For each wave (generate ~30 waves):
    freq  = uniform(0.04, 0.34)
    amp   = uniform(2, 7.5)
    phase = uniform(0, 2π)
    offset = uniform(4, 11)
    length = uniform(60, 100) tokens

    token[t] = clamp(round(offset + amp × sin(2π × freq × t + phase)), 0, 15)
```

### Sampling Training Examples

Each training step, sample a random window from a random wave:

```
wave = random choice from wave corpus
start = random int in [0, len(wave) - 4]
window_len = random int in [3, min(SEQ_LEN, len(wave) - start - 1)]
input = wave[start : start + window_len]
target = wave[start + window_len]
```

### Training Loop

Per tick (runs on ~50ms interval):
1. Sample 24 random (input, target) pairs
2. Run trainStep on each
3. Track average loss
4. Every 300 ticks, regenerate the wave corpus for variety

### Loss Function

Cross-entropy: `loss = -log(probs[target])`

### Backward Pass — CRITICAL

**This is the part that must work correctly.** The previous implementation had bugs in the backward pass that made the model unable to learn (loss stuck at ln(16) ≈ 2.77, which is random guessing).

**Recommended approach: implement BOTH numerical and analytical gradients.**

#### Step 1: Numerical Gradients (guaranteed correct, use for verification)

For each parameter `p`:
```
epsilon = 1e-5
p += epsilon
loss_plus = forward_loss(model, input, target)
p -= 2 * epsilon
loss_minus = forward_loss(model, input, target)
p += epsilon  # restore
gradient = (loss_plus - loss_minus) / (2 * epsilon)
p -= learning_rate * gradient
```

This is O(num_params) forward passes per training step — about 2,160 × 2 = 4,320 forward passes. Each forward pass on a short sequence is very fast in JS, so this should run in ~100-200ms.

**Use numerical gradients as the default training method.** It's slow but correct and the model is small enough.

#### Step 2: Analytical Gradients (optional optimization)

If you want faster training, implement analytical backprop. But **verify it against numerical gradients first**:

```
For each parameter, compute both:
    numerical_grad = (loss(p+ε) - loss(p-ε)) / 2ε
    analytical_grad = your_backprop_result

    relative_error = |numerical - analytical| / max(|numerical|, |analytical|, 1e-8)

    Assert relative_error < 1e-4 for all parameters
```

Run this check on several random inputs before trusting the analytical version. If any parameter fails, the backward pass has a bug.

Key things that are easy to get wrong in the backward pass:
- **Softmax backward**: `d_scores[j] = attn[j] * (d_attn[j] - Σ_k attn[k] * d_attn[k])`, then divide by √d
- **Residual connections**: the gradient flows through both the skip and the transform
- **GELU derivative**: `gelu'(x) = 0.5(1 + tanh(z)) + 0.5*x*(1 - tanh²(z)) * √(2/π) * (1 + 3*0.044715*x²)` where `z = √(2/π)*(x + 0.044715*x³)`
- **Multiple positions contributing to Wk, Wv**: gradients from all attended positions accumulate

#### Gradient Clipping

Clamp all gradients to ±5.0 before applying updates. Skip the backward pass entirely if loss > 20 or is NaN.

#### Learning Rate

Start with 0.01. If loss oscillates or explodes, reduce. The model should show clear loss decrease from ~2.77 within the first 50-100 epochs.

---

## Inference / Generation

Greedy decoding (argmax):
```
next_token = argmax(probs)
```

Do NOT use temperature sampling — for smooth wave continuation, the model's best prediction is what we want.

---

## UI Layout (Phone-First)

Top-to-bottom, full viewport height:

### 1. Header Bar
```
wave.transformer  2,160 params · 16 levels · 12d
```

### 2. Toolbar
```
[⟳ Train] [▶ Play] [◀] [▶] [↺ Rewind] [✕ Clear]     [RESET]
```

- **Train / ■ Stop**: toggles training loop. Red when active.
- **▶ Play / ❚❚ Pause**: starts/pauses autoregressive generation. Play becomes Pause while generating.
- **◀ ▶ Step**: only enabled when paused. Step back undoes last generated token. Step forward generates one token.
- **↺ Rewind**: returns to seed (removes all generated tokens), goes to idle state. Disabled during draw phase.
- **✕ Clear**: clears seed drawing, returns to draw phase. Does NOT reset model weights.
- **RESET**: reinitializes model weights from scratch, clears everything, returns to draw phase.

### 3. Status Bar
```
[○ TRAIL]  seed [====slider====] 24      ep 1234  loss 1.2345  18 gen
```

- **Trail toggle**: on/off. When on, weight textures blend with previous values (exponential moving average, α=0.3) creating a "smear" effect showing change over time.
- **Seed length slider**: range 8–48, adjustable only during draw phase. Controls how many tokens the freehand drawing is quantized into.
- **Status**: epoch count, current loss (4 decimal places), number of generated tokens.

### 4. Wave Canvas (top 25% of viewport, min 120px)

During **draw phase**:
- User draws freely with finger/pointer (continuous line, touch-action: none)
- Raw freehand stroke shown in faded white
- Live preview of quantized version (white dots connected by lines) overlaid
- On pointer-up: path is sampled at `seedSteps` evenly-spaced x positions, y snapped to 0–15 levels
- If the result has ≥3 tokens, transition to idle phase

During **idle / playing / paused**:
- Shows the full sequence: seed portion in white, generated portion in green
- Dashed vertical line at the seed/generation boundary
- Dots at each token position, connected by lines
- Canvas auto-scales horizontally to fit all tokens

Background: horizontal grid lines at each of the 16 levels (every 4th line brighter).

### 5. Texture Panel (bottom 75%, scrollable)

Grid of weight matrix heatmaps, responsive columns (auto-fill, min 140px per cell).

**Static weight textures (always shown, 9 total):**

| Label | Matrix | Shape |
|-------|--------|-------|
| tok embed | tok_emb | 16×12 |
| pos embed | pos_emb | 24×12 |
| W_query | Wq | 12×12 |
| W_key | Wk | 12×12 |
| W_value | Wv | 12×12 |
| W_out | Wo | 12×12 |
| FFN ↑ | ffn_up | 36×12 |
| FFN ↓ | ffn_down | 12×36 |
| readout | out_proj | 16×12 |

**Dynamic textures (shown during generation only):**

| Label | Data | Shape |
|-------|------|-------|
| attention | attention weight matrix | T×T (padded) |
| FFN activations | post-GELU activations per position | T×36 |

**Rendering each texture:**
- Use a `<canvas>` element with `imageRendering: pixelated`
- Cell size: `max(2, min(floor(156/cols), floor(90/rows), 10))` pixels
- Normalize data per-texture: `normalized = ((value - min) / (max - min)) * 2 - 1` mapping to [-1, 1]
- Color map (thermal): blue → green → yellow → red across the -1 to +1 range
- Label below each canvas: `"name shape"` in small monospace text

**Trail mode:**
When trail is enabled, each texture blends with a trail buffer:
```
displayed_value = trail_buffer * 0.7 + current_value * 0.3
trail_buffer = displayed_value  // update for next frame
```

**Probability bar (shown during generation):**
Below the textures, show 16 vertical bars (one per amplitude level), height proportional to the model's output probability for that level. Label each with its level number (0–15).

---

## Interaction State Machine

```
            draw a seed
  [DRAW] ─────────────────→ [IDLE]
    ↑  ↑                      │
    │  │                 Play  │
    │  │                      ↓
    │  │    Pause         [PLAYING]
    │  │  ←─────────────    │
    │  │                    ↓
    │  │               [PAUSED]
    │  │                 │    ↑
    │  │     Step ◀▶     └────┘
    │  │
    │  └── Clear (from any state)
    └───── Reset (from any state)
```

- **DRAW**: canvas accepts pointer input. Only Train and Reset buttons active. Seed slider adjustable.
- **IDLE**: seed is set. Play, Train, Clear, Reset, Rewind active.
- **PLAYING**: model generates tokens on 150ms interval. Pause, Train, Clear, Reset active.
- **PAUSED**: Step ◀▶ enabled. Play, Train, Clear, Reset, Rewind active.

Training can run in any state except DRAW (though it also works in DRAW if the user wants to train before drawing).

---

## Drawing Input Specification

The drawing must feel natural and responsive.

### Capture
- Use `onPointerDown`, `onPointerMove`, `onPointerUp`, `onPointerLeave`
- Set `touch-action: none` on the canvas to prevent scrolling
- Collect raw points as `{x, y}` normalized to 0–1 range within canvas bounds

### Quantization (on pointer-up)
1. Sort collected points by x
2. Find x range (min to max)
3. Sample at `seedSteps` evenly-spaced x positions within that range
4. For each sample point, find the nearest raw point by x distance
5. Convert y to amplitude level: `level = clamp(round((1 - y) * 15), 0, 15)` (y=0 is top of canvas = level 15)
6. Result: array of `seedSteps` integers, each 0–15

### Visual Feedback During Drawing
- Show the raw freehand stroke in semi-transparent white
- Overlay the quantized preview in solid white with dots at each sample point

---

## Technical Constraints

### React Artifact Environment

- Single `.jsx` file, default export is the root component
- Available: `react` (with hooks via `import { useState, ... } from "react"`)
- Available: Tailwind core utility classes only (no compiler)
- **NO localStorage, sessionStorage, or browser storage APIs**
- No external dependencies beyond what's listed in the artifact spec
- All state must live in React state or refs

### Performance

- Training tick: 50ms interval. Must complete batch of 24 training steps within this window.
  - With numerical gradients (~4,320 forward passes per step × 24 steps): this may be too slow. Options:
    - Reduce batch size to 4-8 for numerical gradients
    - Use smaller epsilon neighborhoods
    - Only update a random subset of parameters per step (stochastic coordinate descent)
    - Implement analytical backprop (but verify first!)
  - **Recommended**: get analytical backprop working correctly using numerical gradient verification, then use analytical for training.
- Generation tick: 150ms interval. Single forward pass — trivially fast.
- Canvas rendering: should be fast, using `putImageData` for textures and basic 2D canvas API for the wave.

### Color Palette (thermal heatmap)

```javascript
function heatColor(v) {  // v in [-1, 1]
  const t = (v + 1) / 2;  // map to [0, 1]
  const r = t < 0.5 ? 0 : (t - 0.5) * 2 * 255;
  const g = t < 0.25 ? t * 4 * 255 : t < 0.75 ? 255 : (1 - t) * 4 * 255;
  const b = t < 0.5 ? (0.5 - t) * 2 * 255 : 0;
  return [clamp(r), clamp(g), clamp(b)];
}
```

---

## Acceptance Criteria

1. **Training works**: loss decreases clearly from ~2.77 (random) to below 1.5 within a few hundred epochs. Weight textures visibly develop structure (especially tok_emb should show a gradient pattern — nearby amplitude levels getting similar embeddings).

2. **Generation is smooth**: after training, drawing a gentle sine-like seed and hitting Play should produce a continuation that looks wave-like, not random noise.

3. **Drawing is natural**: freehand drawing on the canvas with finger or pointer feels responsive. The quantized preview updates in real time. Lifting the finger/pointer commits the seed.

4. **All controls work**: Train, Play/Pause, Step ◀▶, Rewind, Clear, Reset, Trail toggle, seed slider all function as specified.

5. **Textures update live**: during both training and generation, the weight heatmaps visibly change. Attention pattern and FFN activations appear during generation.

6. **Phone layout**: works well on a ~390px wide viewport. Canvas in top 25%, textures fill the rest.

---

## Debugging Checklist

If the model isn't learning (loss stuck at ~2.77):

1. **Verify gradients numerically** — this is the #1 diagnostic. Compute `(loss(p+ε) - loss(p-ε)) / 2ε` for every parameter and compare to your analytical gradient. Any relative error > 1e-3 means there's a bug.

2. **Check forward pass** — verify that `forward([0, 1, 2, 3])` produces a valid probability distribution (sums to ~1, no NaN).

3. **Check softmax** — verify it handles large values (subtract max first to avoid overflow).

4. **Check GELU** — verify `gelu(0) ≈ 0`, `gelu(3) ≈ 3`, `gelu(-3) ≈ -0.004`.

5. **Check that training data is correct** — verify generated sine waves actually look sinusoidal when plotted.

6. **Check learning rate** — try values from 0.001 to 0.1. If nothing works, the gradients are wrong.

7. **Try training on a trivial task first** — e.g., `input=[5,5,5,5], target=5`. If the model can't learn to repeat a constant, the backward pass is broken.
