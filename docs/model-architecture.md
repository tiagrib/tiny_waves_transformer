# Wave Transformer — Model Architecture

A tiny autoregressive transformer (GPT-like) that predicts the next amplitude level in a waveform sequence. Instead of text tokens, its vocabulary is 16 amplitude levels (0-15).

## Overview

```
Input tokens: [t0, t1, ..., tN]          (each 0-15)
        |
   [ Embedding ]         token lookup + positional encoding
        |
   [ Self-Attention ]    "what happened before matters"
        |  + residual
   [ Feed-Forward ]      "think about what I saw"
        |  + residual
   [ Output Projection ] predict next token
        |
   probabilities[0..15]  → pick the next amplitude level
```

The model has a single attention head and a single transformer layer. Despite this simplicity, it learns to continue sine-wave-like patterns from a drawn seed.

---

## Hyperparameters

| Name | Value | Notes |
|------|-------|-------|
| `NUM_TOKENS` | 16 | Amplitude levels 0-15 |
| `dim` | configurable (default 24) | Embedding dimension |
| `SEQ_LEN` | 24 | Maximum context window |
| `ffnDim` | dim * 3 | Feed-forward hidden size |
| `NUM_HEADS` | 1 | Single attention head |
| `NUM_LAYERS` | 1 | Single transformer layer |

At dim=12: 2,160 parameters. At dim=24: 7,200 parameters. At dim=48: 26,736 parameters.

---

## Step-by-Step Forward Pass

### Step 1: Token + Positional Embedding

```
embedding[t] = tok_emb[token[t]] + pos_emb[t]
```

**What it does:** Each input token (an integer 0-15) is looked up in a learned embedding table, producing a vector of size `dim`. A separate positional embedding is added so the model knows *where* each token is in the sequence (position 0, 1, 2, ...).

**Why:** Without positional embeddings, the model would see [3, 7, 12] and [12, 7, 3] as identical — it wouldn't know the order. The position signal tells it "3 came first, then 7, then 12."

**Parameters:**
- `tok_emb` — shape [16 x dim]: one row per amplitude level
- `pos_emb` — shape [24 x dim]: one row per position (initialized at small scale 0.1)

### Step 2: Query, Key, Value Projections

```
Q[t] = Wq × embedding[t]    (what am I looking for?)
K[t] = Wk × embedding[t]    (what do I contain?)
V[t] = Wv × embedding[t]    (what do I offer?)
```

**What it does:** Each position's embedding is projected through three separate learned matrices to create three different "views" of the same information:

- **Query (Q):** "I'm at position t, and I want to know about tokens that..."
- **Key (K):** "I'm at position t, and I can tell you about..."
- **Value (V):** "If you attend to me, here's the information I'll give you"

**Why:** This separation lets the model learn *what to look for* independently from *what information to extract*. A token might be a good match (high Q·K score) but carry different useful information (V) than what the query was literally asking about.

**Parameters:**
- `Wq`, `Wk`, `Wv` — each shape [dim x dim]

### Step 3: Causal Self-Attention

```
scores[i][j] = (Q[i] · K[j]) / √dim     for j ≤ i only
weights[i]   = softmax(scores[i])
context[i]   = Σ_j  weights[i][j] × V[j]
```

**What it does:** For each position i, compute how much it should "attend to" every previous position j (including itself). The dot product Q[i]·K[j] measures relevance. Softmax normalizes these into weights that sum to 1. The final context is a weighted average of all the value vectors.

**The causal mask (j ≤ i):** Position 5 can only look at positions 0-5, never at 6+. This is what makes the model *autoregressive* — it can only use past context to predict the future, just like GPT.

**The √dim scaling:** Without this, dot products grow large as dim increases, pushing softmax into saturation (all weight on one token). Dividing by √dim keeps the variance stable.

**Example:** If the model sees [7, 8, 10, 11, 12], position 4 (token 12) might attend heavily to positions 1-3 (the rising pattern 8→10→11) to predict what comes next.

### Step 4: Attention Output + First Residual Connection

```
attn_out[t] = Wo × context[t]
residual_1[t] = embedding[t] + attn_out[t]
```

**What it does:** The context vectors are projected through another learned matrix `Wo`, then added back to the original embeddings.

**The residual connection (+):** This is crucial. It means the model can *either* use the attention information *or* fall back to the original embedding (or mix both). Without it, information from the embedding would have to survive being compressed through attention — the skip connection gives it a direct path through.

**Parameters:**
- `Wo` — shape [dim x dim]

### Step 5: Feed-Forward Network (FFN)

```
ffn_pre[t]  = ffn_up × residual_1[t] + ffn_up_bias     (expand: dim → dim*3)
ffn_act[t]  = GELU(ffn_pre[t])                          (non-linearity)
ffn_out[t]  = ffn_down × ffn_act[t] + ffn_down_bias     (compress: dim*3 → dim)
```

**What it does:** A two-layer neural network applied independently at each position. First it expands the representation to 3x the size, applies a non-linear activation (GELU), then compresses back down.

**Why expand then compress?** The expansion creates a higher-dimensional space where the model can represent more complex patterns. The GELU activation lets it make non-linear decisions (e.g., "if the value is above 8, treat it differently"). The compression projects these decisions back to the working dimension.

**GELU activation:** A smooth approximation to ReLU. Unlike ReLU (which hard-clips at 0), GELU gently dampens negative values: `gelu(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`. At x=3 it's ~3, at x=0 it's 0, at x=-3 it's ~-0.004.

**Parameters:**
- `ffn_up` — shape [dim*3 x dim], `ffn_up_bias` — shape [dim*3]
- `ffn_down` — shape [dim x dim*3], `ffn_down_bias` — shape [dim]

### Step 6: Second Residual Connection

```
residual_2[t] = residual_1[t] + ffn_out[t]
```

**What it does:** Same idea as Step 4 — add the FFN output back to its input. The model can choose to use the FFN's computation or pass information straight through.

### Step 7: Output Projection (Last Token Only)

```
logits = out_proj × residual_2[last]
probs  = softmax(logits)
```

**What it does:** Only the last position's representation is used to predict the next token. It's projected to 16 values (one per amplitude level) and softmax converts these "logits" into a probability distribution.

**Why only the last token?** In autoregressive generation, we only need to predict *one* next token. The last position has attended to all previous positions, so it has the full context. (During training, we still compute all positions because the backward pass needs them.)

**Parameters:**
- `out_proj` — shape [16 x dim]

---

## Parameter Count

At dim=24 (default):

| Matrix | Shape | Params |
|--------|-------|--------|
| tok_emb | 16 x 24 | 384 |
| pos_emb | 24 x 24 | 576 |
| Wq | 24 x 24 | 576 |
| Wk | 24 x 24 | 576 |
| Wv | 24 x 24 | 576 |
| Wo | 24 x 24 | 576 |
| ffn_up | 72 x 24 | 1,728 |
| ffn_up_bias | 72 | 72 |
| ffn_down | 24 x 72 | 1,728 |
| ffn_down_bias | 24 | 24 |
| out_proj | 16 x 24 | 384 |
| **Total** | | **7,200** |

---

## Initialization

- **Weight matrices:** Xavier initialization — `scale = √(2 / (fan_in + fan_out))`. This keeps the variance of activations stable through the network.
- **Positional embeddings:** Scaled down to 0.1 × Xavier. The position signal should be subtle relative to the token content.
- **Biases:** All zeros.
