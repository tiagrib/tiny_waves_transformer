# Tiny Transformers — Demo Guide

All demos share the same 1-layer, 1-head GPT architecture. They differ in what the 16-token vocabulary represents, how training data is generated, and how the user interacts with input and output.

---

## Shared Architecture

Every demo uses the same transformer core from `shared/`:

```
Input tokens (0-15) → Embedding → Self-Attention → FFN → Output probabilities (0-15)
```

The model doesn't know whether it's predicting waveform amplitudes or drum hits — it just learns statistical patterns in sequences of integers 0-15. The "meaning" comes entirely from how each demo encodes and decodes those integers.

### Shared Hyperparameters

| Parameter | Range | Default | Effect (all demos) |
|-----------|-------|---------|-------------------|
| **dim** | 8-48 | 24 | Embedding dimension. Higher = more capacity but slower training. dim=24 gives ~7,200 params, dim=12 gives ~2,160. Requires RESET to apply. |
| **lr** | 0.0001-0.1 | 0.001 | Adam learning rate. Too high → loss oscillates or diverges. Too low → barely learns. 0.001 is the Adam sweet spot for most demos. |
| **temp** | 0-1.5 | 0.0 | Temperature for generation. 0 = always pick the most likely token (deterministic). Higher = sample from the distribution (more variety, more noise). |
| **trail** | on/off | off | Weight texture visualization: blend with previous frame (0.7 old + 0.3 new) to show change over time. |

### How Hyperparameters Affect Each Demo Differently

#### dim (model dimension)

| Demo | dim=12 (2,160 params) | dim=24 (7,200 params) | dim=48 (26,736 params) |
|------|----------------------|----------------------|------------------------|
| **Waves** | Learns rough wave shapes but generates repetitive patterns. Oscillation is choppy. | Good balance — learns smooth oscillation, diverse frequencies. Trains in ~20ms/tick. | Overkill for 30 sine waves. May overfit. Slower (~80ms/tick). |
| **Drums** | Learns basic kick-snare patterns but misses syncopation details. | Captures most rhythm patterns including ghost notes and hi-hat variations. | Can memorize all 30 patterns near-perfectly. Risk of overfitting to specific beats. |

#### lr (learning rate)

| Demo | Effect |
|------|--------|
| **Waves** | Sine waves are smooth, gradual patterns — lr=0.001 works well. Going higher (0.01) causes the loss to oscillate because the model overshoots the smooth gradients. |
| **Drums** | Drum patterns are discrete and repetitive (16-step loops) — the model can tolerate slightly higher lr (0.002-0.003) because the gradient signal is stronger for these categorical patterns. |

#### temp (temperature)

| Demo | temp=0 | temp=0.3-0.5 | temp=0.7-1.0 | temp>1.0 |
|------|--------|--------------|--------------|----------|
| **Waves** | Locked into one repeating oscillation. Deterministic but monotonous. | Slight variation in amplitude — feels more organic. | Noisy waves with occasional jumps. Still wave-like. | Random noise, barely recognizable as waves. |
| **Drums** | Rigid beat loop. Model picks its single best guess for each position. | Occasional fills and variations — most musical setting. | Unstable rhythms with surprising hits. Creative but chaotic. | Random drum noise. |

---

## Demo: Waves

**Vocabulary**: 16 amplitude levels (0 = bottom, 15 = top)

Each token represents a discrete amplitude in a waveform. The sequence is a time series — token at position `t` is the wave's height at time step `t`.

### Training Data

30 quantized sine waves with randomized parameters:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Frequency | 0.04-0.34 | How fast the wave oscillates |
| Amplitude | 4-7.5 | How tall the wave swings |
| Phase | 0-2π | Where in the cycle each wave starts |
| Offset | 5.5-9.5 | Center line of the wave |

These ranges ensure waves span the full 0-15 range. Each wave is 60-100 tokens long, giving diverse training windows.

### User Input

Freehand drawing on a canvas. The continuous stroke is quantized into discrete amplitude tokens at evenly-spaced x positions (controlled by the **seed** slider, 8-48 steps).

### What the Model Learns

- Rising sequences tend to continue rising (until they peak)
- Falling sequences tend to reverse near the extremes
- Nearby amplitude levels (e.g., 7 and 8) are treated as similar
- The general speed of oscillation from the training corpus

### Best Settings

- **dim=24**, **lr=0.001**, train for 1000+ epochs
- **temp=0.3-0.5** for natural-looking wave continuations
- Draw a seed that looks like part of a sine wave for best results

---

## Demo: Drums

**Vocabulary**: 16 tokens = 4-bit bitmask of drum sounds

```
bit 0 (value 1) = Kick drum    (K)
bit 1 (value 2) = Snare drum   (S)
bit 2 (value 4) = Closed hi-hat (H)
bit 3 (value 8) = Open hi-hat  (O)
```

Multiple drums can fire on the same step by combining bits:

| Token | Binary | Drums | Common use |
|-------|--------|-------|------------|
| 0 | 0000 | silence | rests |
| 1 | 0001 | K | downbeat |
| 2 | 0010 | S | backbeat |
| 4 | 0100 | H | time-keeping |
| 5 | 0101 | K+H | downbeat with hat |
| 6 | 0110 | S+H | backbeat with hat |
| 7 | 0111 | K+S+H | accent hit |
| 8 | 1000 | O | open hat for flavor |

### Training Data

30 drum patterns — 8 hand-crafted classics + 22 procedurally generated:

**Hand-crafted patterns:**
1. **4-on-the-floor** (house/disco) — kick every beat, hat on every step
2. **Standard rock** — kick on 1 & 3, snare on 2 & 4, hats on 8ths
3. **Funk** — syncopated kick, ghost snares
4. **Reggaeton/dembow** — signature K-S-K-S pattern
5. **Bossa nova** — off-beat accents
6. **Breakbeat** — fragmented, sample-chopped feel
7-8. **Trap/hi-hat focused** — dense hat patterns

**Procedural generation** combines:
- 3 kick styles: 4-on-the-floor, 1&3, syncopated
- 3 snare styles: backbeat, ghost notes, offbeat
- 3 hi-hat styles: 8ths, 16ths, sparse/offbeat
- Optional open hats replacing closed hats

Each 16-step pattern is repeated 4x (64 steps) for varied training window positions.

### User Input

A 4-row × 16-step beat grid. Each row is one drum sound (K/S/H/O), each column is one time step. Tap cells to toggle hits on/off, then press "Set Seed."

### Audio

Web Audio synthesis — no samples needed:
- **Kick**: Sine oscillator with frequency sweep 150Hz→40Hz (punchy thump)
- **Snare**: White noise burst + triangle body tone (snappy crack)
- **Closed hat**: Short filtered noise (tight tick)
- **Open hat**: Longer filtered noise (ringy wash)

Playback speed is controlled by the **bpm** slider (60-200). Each step is a 16th note.

### Drums-Specific Hyperparameter

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **bpm** | 60-200 | 120 | Playback tempo in beats per minute. Each step is a 16th note, so 120 BPM = 8 steps/second. Does not affect training. |

### What the Model Learns

- Kick drums tend to land on strong beats (positions 0, 4, 8, 12)
- Snares follow kicks on backbeats (positions 4, 12)
- Hi-hats fill the space between kick and snare
- Open hats are rare and appear in specific positions
- The 16-step loop structure (patterns tend to repeat or rhyme every 16 steps)

### Best Settings

- **dim=24**, **lr=0.001**, train for 500-1000 epochs (drums converge faster than waves due to stronger patterns)
- **temp=0.3-0.5** for natural beat variations with occasional fills
- Start with a simple seed (kick-hat-snare-hat) for the model to build on

---

## Adding a New Demo

To create a new demo using the shared transformer:

1. **Choose a vocabulary mapping** for the 16 tokens (what does each integer 0-15 mean?)
2. **Create `demos/{name}/data.js`** with:
   - A function to generate training sequences (like `generateWaves` or `generateDrumPatterns`)
   - A `sampleExample(data, seqLen)` function that returns `{ input, target }` pairs
3. **Create `demos/{name}/Demo.jsx`** with:
   - A React component (PascalCase name) that uses the shared functions as globals
   - Demo-specific UI for seed input and output visualization
4. **Run `node build.js`** — the build script auto-discovers and builds it
5. **Add a card** to `demos/index.html`

The shared functions available to every demo:

| From | Functions |
|------|-----------|
| `shared/core.js` | `zeros`, `xavierInit`, `matVec`, `vecAdd`, `dot`, `softmax`, `gelu`, `geluDerivative` |
| `shared/model.js` | `createModel({config})`, `forward`, `backward`, `countParams`, `crossEntropyLoss` |
| `shared/optimizer.js` | `createAdam`, `adamStep`, `trainStep`, `clipGrad` |
| `shared/ui.js` | `heatColor`, `renderTexture`, `sampleWithTemp` |
