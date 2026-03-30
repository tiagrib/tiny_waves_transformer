# Wave Transformer — UI and Visualization

## Interaction Flow

```
  [DRAW] ──draw seed──→ [IDLE] ──Play──→ [PLAYING] ←──→ [PAUSED]
    ↑                                                     Step ◀▶
    └──── Clear (from any state) / Reset (from any state) ────┘
```

Training can run in any state — it's independent of the draw/play lifecycle.

## Controls

| Button | What it does |
|--------|-------------|
| **Train / Stop** | Toggles the training loop (24 samples per 50ms tick) |
| **Play / Pause** | Starts/pauses autoregressive generation (one token per 150ms) |
| **Step ◀** | (Paused only) Remove last generated token |
| **Step ▶** | (Paused only) Generate one token |
| **Rewind** | Remove all generated tokens, keep seed, return to IDLE |
| **Clear** | Remove seed and generated tokens, return to DRAW phase |
| **RESET** | Reinitialize model weights, clear everything, start fresh |

## Sliders

| Slider | Range | Default | Notes |
|--------|-------|---------|-------|
| **seed** | 8-48 | 24 | Number of tokens the drawing is quantized into (DRAW phase only) |
| **dim** | 8-48 | 24 | Model embedding dimension. Requires RESET to take effect |
| **lr** | 0.0001-0.1 | 0.001 | Learning rate (logarithmic scale). Adjustable during training |
| **temp** | 0-1.5 | 0.0 | Temperature for generation sampling. 0 = greedy/deterministic |

## Wave Canvas (top 25%)

### During DRAW phase (no seed yet)

If training is active, shows a **live loss graph**:
- Orange curve showing loss over epochs
- Y-axis labels with loss values
- X-axis showing epoch count
- Scales from the initial loss peak (~2.77) down to current minimum

If not training, shows "draw a seed wave" prompt.

When the user draws (pointer down + move), shows:
- **Faded white line:** Raw freehand stroke
- **Solid white dots + lines:** Quantized preview (snapped to amplitude levels)

On pointer up, if the quantized result has 3+ tokens, it becomes the seed and transitions to IDLE.

### During IDLE / PLAYING / PAUSED

- **White dots + line:** Seed portion
- **Green dots + line:** Generated portion
- **Dashed vertical line:** Boundary between seed and generated
- **Gray horizontal lines:** Grid at each of the 16 amplitude levels (every 4th line brighter)

Canvas auto-scales horizontally to fit all tokens.

## Drawing Input

1. User draws freely with pointer/finger (touch-action: none prevents scrolling)
2. Raw points are collected as normalized {x, y} coordinates (0-1 range)
3. On pointer up, points are quantized:
   - Sorted by x
   - Sampled at `seedSteps` evenly-spaced x positions
   - Y mapped to amplitude: `level = clamp(round((1 - y) * 15), 0, 15)`
   - Top of canvas = level 15, bottom = level 0

## Weight Textures (bottom 75%)

A grid of heatmap visualizations of every weight matrix in the model. Responsive columns (auto-fill, min 140px each).

### Static textures (always shown, 9 total)

| Label | Matrix | What it represents |
|-------|--------|--------------------|
| tok embed | tok_emb [16 x dim] | How each amplitude level is represented |
| pos embed | pos_emb [24 x dim] | How each position is encoded |
| W_query | Wq [dim x dim] | Query projection for attention |
| W_key | Wk [dim x dim] | Key projection for attention |
| W_value | Wv [dim x dim] | Value projection for attention |
| W_out | Wo [dim x dim] | Attention output projection |
| FFN up | ffn_up [dim*3 x dim] | FFN expansion weights |
| FFN down | ffn_down [dim x dim*3] | FFN compression weights |
| readout | out_proj [16 x dim] | Output prediction weights |

### Dynamic textures (shown during generation only)

| Label | Data | What it shows |
|-------|------|---------------|
| attention | [T x T] attention weights | How much each position attends to previous positions |
| FFN act | [T x dim*3] post-GELU activations | Which FFN neurons fired for each position |

### Rendering

Each texture is rendered as a `<canvas>` with `imageRendering: pixelated`:
- Cell size adapts to matrix dimensions
- Values are normalized per-texture to [-1, 1]: `(value - min) / (max - min) * 2 - 1`
- Color map (thermal): blue (-1) → green (0) → yellow → red (+1)

### Trail mode

When the TRAIL toggle is on, textures blend with their previous values:
```
displayed = old_buffer × 0.7 + current × 0.3
```
This creates a smear effect that shows how weights change over time during training.

## Probability Bar (during generation)

Below the textures, 16 vertical bars show the model's output probability distribution for the most recent generation step. Each bar corresponds to one amplitude level (0-15), with height proportional to probability.

## Header

Dynamically shows: `wave.transformer {param_count} params · 16 levels · {dim}d`
