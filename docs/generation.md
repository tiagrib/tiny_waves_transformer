# Wave Transformer — Generation (Inference)

How the trained model continues a wave from a user-drawn seed.

## Autoregressive Generation

The model generates one token at a time, feeding each prediction back as input:

```
seed:      [7, 8, 10, 11, 12]     ← user drew this
                                    ↓ forward pass
                              model predicts → 11
           [7, 8, 10, 11, 12, 11]
                                    ↓ forward pass
                              model predicts → 9
           [7, 8, 10, 11, 12, 11, 9]
                                    ↓ forward pass
                              model predicts → 7
           ...and so on
```

Each generation step:
1. Take the last 24 tokens (SEQ_LEN) as context
2. Run a full forward pass
3. Get the probability distribution over 16 amplitude levels
4. Pick the next token (greedy or temperature sampling)
5. Append to the sequence
6. Repeat every 150ms

## Token Selection: Temperature

The `temperature` parameter controls how the next token is chosen from the model's output probabilities.

### Temperature = 0 (Greedy / Argmax)

Pick the single most likely token:

```
next = argmax(probs)
```

This is deterministic — the same seed always produces the same continuation. Tends to produce repetitive, locked-in patterns because the model always takes its "safest" bet.

### Temperature > 0 (Sampling)

Scale the logits (pre-softmax scores) before converting to probabilities:

```
scaled_logits[i] = logits[i] / temperature
probs = softmax(scaled_logits)
next = random sample from probs
```

**Low temperature (0.1-0.5):** Makes the distribution "sharper" — the top token gets even more probability. Mostly follows the greedy path but occasionally varies.

**Temperature = 1.0:** Uses the model's raw probability distribution as-is. The model's true uncertainty is reflected in the sampling.

**High temperature (>1.0):** Makes the distribution "flatter" — all tokens become more equally likely. Produces diverse but potentially noisy output.

### Example

If the model outputs logits `[3.0, 2.5, 1.0, ...]` for a position:

| Temperature | P(token 0) | P(token 1) | P(token 2) | Behavior |
|-------------|-----------|-----------|-----------|----------|
| 0 (greedy)  | picked    | —         | —         | Always token 0 |
| 0.5         | 88%       | 10%       | 1%        | Almost always 0 |
| 1.0         | 54%       | 33%       | 7%        | Often 0, sometimes 1 |
| 1.5         | 40%       | 30%       | 13%       | Varied, occasionally wild |

### Recommended values

- **temp = 0:** Best for seeing what the model "really learned" — deterministic, clean
- **temp = 0.3-0.5:** Good balance — mostly follows learned patterns with slight variation
- **temp = 0.7-1.0:** More natural-looking waves, less repetitive
- **temp > 1.0:** Experimental, noisy output

## Context Window

The model can only see the last 24 tokens (SEQ_LEN). When the sequence grows longer than 24, older tokens are dropped:

```
Full sequence: [7, 8, 10, 11, 12, 11, 9, 7, 5, 3, 2, 3, 5, 7, 9, 11, 12, 12, 11, 9, 7, 5, 3, 2, 1, 2, 4]
Context used:                                   [5, 3, 2, 3, 5, 7, 9, 11, 12, 12, 11, 9, 7, 5, 3, 2, 1, 2, 4]
                                                 └──────────────────── last 24 tokens ────────────────────────┘
```

This means the model "forgets" the original seed after enough tokens are generated. The character of the continuation depends on recent context only.

## What the Model Learns

Trained on 30 quantized sine waves, the model learns:

- **Local patterns:** Rising sequences tend to keep rising (up to a peak). Falling sequences tend to keep falling.
- **Oscillation:** After reaching an extreme (near 0 or 15), the direction reverses.
- **Frequency memory:** The speed of oscillation roughly matches what it saw in training.
- **Amplitude levels:** Token embedding similarity — nearby levels (e.g., 7 and 8) are more similar than distant ones (7 and 14).

It does NOT learn a perfect sine function. It's pattern-matching from ~2000 training examples, approximating wave-like behavior with a 7,200-parameter model.
