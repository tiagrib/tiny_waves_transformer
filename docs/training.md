# Wave Transformer — Training

How the model learns to predict waveforms, from data generation through backpropagation and optimization.

## Training Data

### Wave Corpus Generation

The model trains on 30 randomly generated quantized sine waves:

```
For each wave:
    freq   = uniform(0.04, 0.34)      how fast it oscillates
    amp    = uniform(4, 7.5)           how tall the wave is
    phase  = uniform(0, 2π)            where in the cycle it starts
    offset = uniform(5.5, 9.5)         center line of the wave
    length = uniform(60, 100) tokens

    token[t] = clamp(round(offset + amp × sin(2π × freq × t + phase)), 0, 15)
```

The offset and amplitude ranges are chosen so waves span the full 0-15 range. With offset centered around 7.5 and amplitude up to 7.5, waves can swing from 0 to 15.

### Sampling Training Examples

Each training step, a random window is cut from a random wave:

```
wave   = random choice from the 30 waves
start  = random position in the wave
length = random 3 to 24 tokens (up to SEQ_LEN)
input  = wave[start : start + length]
target = wave[start + length]           ← the next token to predict
```

Variable-length windows teach the model to work with any amount of context, from just 3 tokens up to the full 24-token window.

---

## Loss Function

**Cross-entropy loss** measures how surprised the model is by the correct answer:

```
loss = -log(probs[target])
```

- If the model assigns probability 1.0 to the correct token: loss = 0 (perfect)
- If the model assigns probability 1/16 (random guessing): loss = ln(16) ≈ 2.77
- If the model assigns probability 0.0 to the correct token: loss = ∞ (catastrophic)

So loss starts near 2.77 (random) and decreases as the model learns. A well-trained model at dim=24 reaches loss ~1.5-2.0 on 30 diverse waves.

---

## Backward Pass (Backpropagation)

The backward pass computes the gradient of the loss with respect to every parameter — i.e., "how should I adjust this parameter to reduce the loss?"

It works in reverse order through the forward pass (chain rule):

### Step 1: Output gradient

```
dLogits[i] = probs[i] - (i == target ? 1 : 0)
```

This is the combined gradient of softmax + cross-entropy. Elegantly simple: the gradient is just "what the model predicted minus what it should have predicted."

### Step 2: Backprop through output projection

```
d_out_proj[i][j]  = dLogits[i] × residual_2[last][j]
d_residual_2[last] += out_proj^T × dLogits
```

### Step 3: Backprop through FFN + residual

For each position (in any order, since positions are independent in FFN):

```
d_residual_1 += d_residual_2              (residual skip)
d_ffn_out     = d_residual_2              (residual branch)
    ↓
d_ffn_down, d_ffn_act  ← from ffn_out = ffn_down × ffn_act + bias
    ↓
d_ffn_pre = d_ffn_act × gelu'(ffn_pre)   (chain rule through GELU)
    ↓
d_ffn_up, d_residual_1 ← from ffn_pre = ffn_up × residual_1 + bias
```

### Step 4: Backprop through attention

This is the tricky part. Gradients flow:
1. Through the residual: `d_embedding += d_residual_1`
2. Through `Wo`: `d_context = Wo^T × d_attn_out`
3. Through softmax attention weights → scores
4. Through Q, K, V projections back to embeddings

The softmax backward is: `d_scores[j] = attn[j] × (d_attn[j] - Σ_k attn[k] × d_attn[k])`, then divided by √dim.

Key subtlety: K and V gradients accumulate from *every position that attended to them*. Position 0's key/value are attended to by all later positions, so its gradient is a sum of contributions from positions 0, 1, 2, ..., T-1.

### Step 5: Backprop to embeddings

```
d_tok_emb[token[t]] += d_embedding[t]
d_pos_emb[t]        += d_embedding[t]
```

### Gradient Clipping

All gradients are clamped to ±5.0 before being applied. This prevents explosive updates when the model encounters unusual inputs. If the loss exceeds 20 or is NaN, the entire backward pass is skipped.

---

## Adam Optimizer

Instead of vanilla SGD (`param -= lr × gradient`), we use Adam, which maintains two running averages per parameter:

```
m = β1 × m + (1 - β1) × gradient        first moment (momentum)
v = β2 × v + (1 - β2) × gradient²       second moment (RMS)

m_hat = m / (1 - β1^t)                   bias correction
v_hat = v / (1 - β2^t)                   bias correction

param -= lr × m_hat / (√v_hat + ε)
```

**Why Adam over SGD?**
- **Momentum (m):** Smooths out noisy gradients. If the gradient consistently points one direction, momentum accelerates. If it oscillates, momentum dampens.
- **Adaptive rates (v):** Parameters with large gradients get smaller effective learning rates. Parameters with small gradients get larger ones. This means the model automatically adjusts step sizes per-parameter.
- **Bias correction:** In early steps, m and v are biased toward zero (since they start at 0). The correction compensates for this warmup period.

**Hyperparameters:**
- `β1 = 0.9` — momentum decay (looks back ~10 steps)
- `β2 = 0.999` — RMS decay (looks back ~1000 steps)
- `ε = 1e-8` — prevents division by zero
- `lr = 0.001` — default learning rate (adjustable via slider)

---

## Training Loop

Each training tick (every 50ms):
1. Sample 24 random (input, target) pairs from the wave corpus
2. For each pair: forward pass → compute loss → backward pass → Adam update
3. Track average loss and update the UI

At dim=24 with Adam, each tick takes ~20ms, well within the 50ms budget.
