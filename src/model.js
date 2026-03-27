// Wave Transformer — Core Model
// A tiny autoregressive transformer for waveform generation.
// Vocabulary: 16 amplitude levels (0–15), single-head single-layer GPT.

const NUM_TOKENS = 16;
const EMBED_DIM = 12;
const SEQ_LEN = 24;
const FFN_DIM = 36;
const NUM_HEADS = 1;
const NUM_LAYERS = 1;

// --- Linear Algebra Utilities ---

function zeros(rows, cols) {
  if (cols === undefined) return new Float64Array(rows);
  const m = [];
  for (let i = 0; i < rows; i++) m.push(new Float64Array(cols));
  return m;
}

function xavierInit(rows, cols) {
  const scale = Math.sqrt(2.0 / (rows + cols));
  const m = [];
  for (let i = 0; i < rows; i++) {
    const row = new Float64Array(cols);
    for (let j = 0; j < cols; j++) {
      row[j] = (Math.random() * 2 - 1) * scale;
    }
    m.push(row);
  }
  return m;
}

function matVec(mat, vec) {
  const rows = mat.length;
  const cols = vec.length;
  const out = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    let s = 0;
    for (let j = 0; j < cols; j++) s += mat[i][j] * vec[j];
    out[i] = s;
  }
  return out;
}

function vecAdd(a, b) {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i] - max);
    sum += exps[i];
  }
  for (let i = 0; i < logits.length; i++) exps[i] /= sum;
  return exps;
}

function gelu(x) {
  const c = Math.sqrt(2.0 / Math.PI);
  const inner = c * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1.0 + Math.tanh(inner));
}

function geluDerivative(x) {
  const c = Math.sqrt(2.0 / Math.PI);
  const x3 = x * x * x;
  const z = c * (x + 0.044715 * x3);
  const tanhZ = Math.tanh(z);
  const sech2Z = 1.0 - tanhZ * tanhZ;
  return 0.5 * (1.0 + tanhZ) + 0.5 * x * sech2Z * c * (1.0 + 3.0 * 0.044715 * x * x);
}

// --- Model Creation ---

function createModel() {
  return {
    tok_emb: xavierInit(NUM_TOKENS, EMBED_DIM),
    pos_emb: (() => {
      const m = xavierInit(SEQ_LEN, EMBED_DIM);
      for (let i = 0; i < SEQ_LEN; i++)
        for (let j = 0; j < EMBED_DIM; j++)
          m[i][j] *= 0.1 / Math.sqrt(2.0 / (SEQ_LEN + EMBED_DIM));
      return m;
    })(),
    Wq: xavierInit(EMBED_DIM, EMBED_DIM),
    Wk: xavierInit(EMBED_DIM, EMBED_DIM),
    Wv: xavierInit(EMBED_DIM, EMBED_DIM),
    Wo: xavierInit(EMBED_DIM, EMBED_DIM),
    ffn_up: xavierInit(FFN_DIM, EMBED_DIM),
    ffn_up_bias: new Float64Array(FFN_DIM),
    ffn_down: xavierInit(EMBED_DIM, FFN_DIM),
    ffn_down_bias: new Float64Array(EMBED_DIM),
    out_proj: xavierInit(NUM_TOKENS, EMBED_DIM),
  };
}

// --- Forward Pass ---
// Returns { probs, cache } where cache contains intermediates for backward pass.

function forward(model, tokens) {
  const T = tokens.length;
  const sqrtD = Math.sqrt(EMBED_DIM);

  // Embedding: tok_emb[token] + pos_emb[position]
  const embeddings = [];
  for (let t = 0; t < T; t++) {
    embeddings.push(vecAdd(model.tok_emb[tokens[t]], model.pos_emb[t]));
  }

  // Q, K, V projections
  const Q = [], K = [], V = [];
  for (let t = 0; t < T; t++) {
    Q.push(matVec(model.Wq, embeddings[t]));
    K.push(matVec(model.Wk, embeddings[t]));
    V.push(matVec(model.Wv, embeddings[t]));
  }

  // Causal self-attention
  const attnWeights = []; // T x T (ragged: row i has i+1 entries)
  const contexts = [];
  for (let i = 0; i < T; i++) {
    // Compute scores for positions 0..i
    const scores = new Float64Array(i + 1);
    for (let j = 0; j <= i; j++) {
      scores[j] = dot(Q[i], K[j]) / sqrtD;
    }
    const weights = softmax(scores);
    attnWeights.push(weights);

    // Weighted sum of values
    const ctx = new Float64Array(EMBED_DIM);
    for (let j = 0; j <= i; j++) {
      for (let d = 0; d < EMBED_DIM; d++) {
        ctx[d] += weights[j] * V[j][d];
      }
    }
    contexts.push(ctx);
  }

  // Attention output projection + residual
  const attnOut = [];
  const residual1 = [];
  for (let t = 0; t < T; t++) {
    const ao = matVec(model.Wo, contexts[t]);
    attnOut.push(ao);
    residual1.push(vecAdd(embeddings[t], ao));
  }

  // FFN
  const ffnPre = [];  // before GELU
  const ffnAct = [];  // after GELU
  const ffnOut = [];
  const residual2 = [];
  for (let t = 0; t < T; t++) {
    const pre = vecAdd(matVec(model.ffn_up, residual1[t]), model.ffn_up_bias);
    ffnPre.push(pre);

    const act = new Float64Array(FFN_DIM);
    for (let j = 0; j < FFN_DIM; j++) act[j] = gelu(pre[j]);
    ffnAct.push(act);

    const out = vecAdd(matVec(model.ffn_down, act), model.ffn_down_bias);
    ffnOut.push(out);
    residual2.push(vecAdd(residual1[t], out));
  }

  // Output projection (only last token)
  const logits = matVec(model.out_proj, residual2[T - 1]);
  const probs = softmax(logits);

  return {
    probs,
    cache: {
      tokens, T, embeddings, Q, K, V, attnWeights, contexts,
      attnOut, residual1, ffnPre, ffnAct, ffnOut, residual2, logits,
    },
  };
}

// --- Loss ---

function crossEntropyLoss(probs, target) {
  return -Math.log(Math.max(probs[target], 1e-12));
}

// --- Data Generation ---

function generateWaves(count) {
  const waves = [];
  for (let i = 0; i < count; i++) {
    const freq = 0.04 + Math.random() * 0.30;
    const amp = 2 + Math.random() * 5.5;
    const phase = Math.random() * 2 * Math.PI;
    const offset = 4 + Math.random() * 7;
    const length = Math.floor(60 + Math.random() * 41);
    const wave = [];
    for (let t = 0; t < length; t++) {
      const val = offset + amp * Math.sin(2 * Math.PI * freq * t + phase);
      wave.push(Math.max(0, Math.min(15, Math.round(val))));
    }
    waves.push(wave);
  }
  return waves;
}

function sampleTrainingExample(waves) {
  const wave = waves[Math.floor(Math.random() * waves.length)];
  const maxStart = wave.length - 4;
  const start = Math.floor(Math.random() * (maxStart + 1));
  const maxLen = Math.min(SEQ_LEN, wave.length - start - 1);
  const windowLen = 3 + Math.floor(Math.random() * (maxLen - 2));
  const input = wave.slice(start, start + windowLen);
  const target = wave[start + windowLen];
  return { input, target };
}

// --- Backward Pass (Analytical) ---

function backward(model, cache, target) {
  const { tokens, T, embeddings, Q, K, V, attnWeights, contexts,
    attnOut, residual1, ffnPre, ffnAct, ffnOut, residual2, logits } = cache;
  const sqrtD = Math.sqrt(EMBED_DIM);

  // Initialize gradients
  const grads = {
    tok_emb: zeros(NUM_TOKENS, EMBED_DIM),
    pos_emb: zeros(SEQ_LEN, EMBED_DIM),
    Wq: zeros(EMBED_DIM, EMBED_DIM),
    Wk: zeros(EMBED_DIM, EMBED_DIM),
    Wv: zeros(EMBED_DIM, EMBED_DIM),
    Wo: zeros(EMBED_DIM, EMBED_DIM),
    ffn_up: zeros(FFN_DIM, EMBED_DIM),
    ffn_up_bias: new Float64Array(FFN_DIM),
    ffn_down: zeros(EMBED_DIM, FFN_DIM),
    ffn_down_bias: new Float64Array(EMBED_DIM),
    out_proj: zeros(NUM_TOKENS, EMBED_DIM),
  };

  // d_loss/d_logits: softmax + cross-entropy gradient
  const probs = softmax(logits);
  const dLogits = new Float64Array(NUM_TOKENS);
  for (let i = 0; i < NUM_TOKENS; i++) {
    dLogits[i] = probs[i] - (i === target ? 1 : 0);
  }

  // d_out_proj and d_residual2[T-1]
  const dResidual2 = [];
  for (let t = 0; t < T; t++) dResidual2.push(new Float64Array(EMBED_DIM));

  // out_proj gradient: dLogits is [NUM_TOKENS], residual2[T-1] is [EMBED_DIM]
  // logits = out_proj * residual2[T-1]
  for (let i = 0; i < NUM_TOKENS; i++) {
    for (let j = 0; j < EMBED_DIM; j++) {
      grads.out_proj[i][j] = dLogits[i] * residual2[T - 1][j];
    }
  }
  // d_residual2[T-1] = out_proj^T * dLogits
  for (let j = 0; j < EMBED_DIM; j++) {
    for (let i = 0; i < NUM_TOKENS; i++) {
      dResidual2[T - 1][j] += model.out_proj[i][j] * dLogits[i];
    }
  }

  // Backward through FFN + residual for each position
  const dResidual1 = [];
  for (let t = 0; t < T; t++) dResidual1.push(new Float64Array(EMBED_DIM));

  for (let t = 0; t < T; t++) {
    // residual2 = residual1 + ffnOut => dResidual1 += dResidual2, dFfnOut = dResidual2
    const dFfnOut = new Float64Array(dResidual2[t]);
    for (let j = 0; j < EMBED_DIM; j++) dResidual1[t][j] += dResidual2[t][j];

    // ffnOut = ffn_down * ffnAct + ffn_down_bias
    // d_ffn_down_bias
    for (let j = 0; j < EMBED_DIM; j++) grads.ffn_down_bias[j] += dFfnOut[j];

    // d_ffn_down[i][j] += dFfnOut[i] * ffnAct[t][j]
    const dFfnAct = new Float64Array(FFN_DIM);
    for (let i = 0; i < EMBED_DIM; i++) {
      for (let j = 0; j < FFN_DIM; j++) {
        grads.ffn_down[i][j] += dFfnOut[i] * ffnAct[t][j];
        dFfnAct[j] += model.ffn_down[i][j] * dFfnOut[i];
      }
    }

    // Through GELU: dFfnPre[j] = dFfnAct[j] * gelu'(ffnPre[t][j])
    const dFfnPre = new Float64Array(FFN_DIM);
    for (let j = 0; j < FFN_DIM; j++) {
      dFfnPre[j] = dFfnAct[j] * geluDerivative(ffnPre[t][j]);
    }

    // ffnPre = ffn_up * residual1 + ffn_up_bias
    for (let j = 0; j < FFN_DIM; j++) grads.ffn_up_bias[j] += dFfnPre[j];

    for (let i = 0; i < FFN_DIM; i++) {
      for (let j = 0; j < EMBED_DIM; j++) {
        grads.ffn_up[i][j] += dFfnPre[i] * residual1[t][j];
        dResidual1[t][j] += model.ffn_up[i][j] * dFfnPre[i];
      }
    }
  }

  // Backward through attention + residual
  const dEmbeddings = [];
  for (let t = 0; t < T; t++) dEmbeddings.push(new Float64Array(EMBED_DIM));

  // residual1 = embeddings + attnOut => dEmbeddings += dResidual1, dAttnOut = dResidual1
  const dAttnOut = [];
  for (let t = 0; t < T; t++) {
    dAttnOut.push(new Float64Array(dResidual1[t]));
    for (let j = 0; j < EMBED_DIM; j++) dEmbeddings[t][j] += dResidual1[t][j];
  }

  // attnOut[t] = Wo * contexts[t]
  const dContexts = [];
  for (let t = 0; t < T; t++) {
    dContexts.push(new Float64Array(EMBED_DIM));
    for (let i = 0; i < EMBED_DIM; i++) {
      for (let j = 0; j < EMBED_DIM; j++) {
        grads.Wo[i][j] += dAttnOut[t][i] * contexts[t][j];
        dContexts[t][j] += model.Wo[i][j] * dAttnOut[t][i];
      }
    }
  }

  // context[i] = sum_j attnWeights[i][j] * V[j]
  // d_attnWeights[i][j] = dot(dContexts[i], V[j])
  // d_V[j] += attnWeights[i][j] * dContexts[i]  (for all i >= j)
  const dQ = [];
  const dK = [];
  const dV = [];
  for (let t = 0; t < T; t++) {
    dQ.push(new Float64Array(EMBED_DIM));
    dK.push(new Float64Array(EMBED_DIM));
    dV.push(new Float64Array(EMBED_DIM));
  }

  for (let i = 0; i < T; i++) {
    // d_attnWeights[i][j]
    const dWeights = new Float64Array(i + 1);
    for (let j = 0; j <= i; j++) {
      dWeights[j] = dot(dContexts[i], V[j]);
      // accumulate dV
      for (let d = 0; d < EMBED_DIM; d++) {
        dV[j][d] += attnWeights[i][j] * dContexts[i][d];
      }
    }

    // Softmax backward: d_scores[j] = attn[j] * (d_attn[j] - sum_k attn[k]*d_attn[k])
    let weightedSum = 0;
    for (let j = 0; j <= i; j++) {
      weightedSum += attnWeights[i][j] * dWeights[j];
    }
    const dScores = new Float64Array(i + 1);
    for (let j = 0; j <= i; j++) {
      dScores[j] = attnWeights[i][j] * (dWeights[j] - weightedSum);
    }

    // scores[i][j] = dot(Q[i], K[j]) / sqrtD
    // d_Q[i] += sum_j dScores[j] * K[j] / sqrtD
    // d_K[j] += dScores[j] * Q[i] / sqrtD
    for (let j = 0; j <= i; j++) {
      const s = dScores[j] / sqrtD;
      for (let d = 0; d < EMBED_DIM; d++) {
        dQ[i][d] += s * K[j][d];
        dK[j][d] += s * Q[i][d];
      }
    }
  }

  // Q[t] = Wq * embeddings[t], etc.
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < EMBED_DIM; i++) {
      for (let j = 0; j < EMBED_DIM; j++) {
        grads.Wq[i][j] += dQ[t][i] * embeddings[t][j];
        grads.Wk[i][j] += dK[t][i] * embeddings[t][j];
        grads.Wv[i][j] += dV[t][i] * embeddings[t][j];
        dEmbeddings[t][j] += model.Wq[i][j] * dQ[t][i];
        dEmbeddings[t][j] += model.Wk[i][j] * dK[t][i];
        dEmbeddings[t][j] += model.Wv[i][j] * dV[t][i];
      }
    }
  }

  // embeddings[t] = tok_emb[tokens[t]] + pos_emb[t]
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < EMBED_DIM; d++) {
      grads.tok_emb[tokens[t]][d] += dEmbeddings[t][d];
      grads.pos_emb[t][d] += dEmbeddings[t][d];
    }
  }

  return grads;
}

// --- Training Step ---

function clipGrad(val) {
  return Math.max(-5.0, Math.min(5.0, val));
}

const MATRIX_PARAMS = ['tok_emb', 'pos_emb', 'Wq', 'Wk', 'Wv', 'Wo', 'ffn_up', 'ffn_down', 'out_proj'];
const BIAS_PARAMS = ['ffn_up_bias', 'ffn_down_bias'];

function createGradAccumulator() {
  return {
    tok_emb: zeros(NUM_TOKENS, EMBED_DIM),
    pos_emb: zeros(SEQ_LEN, EMBED_DIM),
    Wq: zeros(EMBED_DIM, EMBED_DIM),
    Wk: zeros(EMBED_DIM, EMBED_DIM),
    Wv: zeros(EMBED_DIM, EMBED_DIM),
    Wo: zeros(EMBED_DIM, EMBED_DIM),
    ffn_up: zeros(FFN_DIM, EMBED_DIM),
    ffn_up_bias: new Float64Array(FFN_DIM),
    ffn_down: zeros(EMBED_DIM, FFN_DIM),
    ffn_down_bias: new Float64Array(EMBED_DIM),
    out_proj: zeros(NUM_TOKENS, EMBED_DIM),
  };
}

function accumulateGrads(acc, grads) {
  for (const name of MATRIX_PARAMS) {
    const a = acc[name], g = grads[name];
    for (let i = 0; i < a.length; i++)
      for (let j = 0; j < a[i].length; j++)
        a[i][j] += g[i][j];
  }
  for (const name of BIAS_PARAMS) {
    const a = acc[name], g = grads[name];
    for (let i = 0; i < a.length; i++) a[i] += g[i];
  }
}

function applyGrads(model, acc, lr, batchSize) {
  const scale = lr / batchSize;
  for (const name of MATRIX_PARAMS) {
    const mat = model[name], grad = acc[name];
    for (let i = 0; i < mat.length; i++)
      for (let j = 0; j < mat[i].length; j++)
        mat[i][j] -= scale * clipGrad(grad[i][j]);
  }
  for (const name of BIAS_PARAMS) {
    const bias = model[name], grad = acc[name];
    for (let i = 0; i < bias.length; i++)
      bias[i] -= scale * clipGrad(grad[i]);
  }
}

function trainStep(model, input, target, lr) {
  const { probs, cache } = forward(model, input);
  const loss = crossEntropyLoss(probs, target);
  if (loss > 20 || isNaN(loss)) return loss;

  const grads = backward(model, cache, target);

  // Apply gradients with clipping
  for (const name of MATRIX_PARAMS) {
    const mat = model[name];
    const grad = grads[name];
    for (let i = 0; i < mat.length; i++) {
      for (let j = 0; j < mat[i].length; j++) {
        mat[i][j] -= lr * clipGrad(grad[i][j]);
      }
    }
  }
  for (const name of BIAS_PARAMS) {
    const bias = model[name];
    const grad = grads[name];
    for (let i = 0; i < bias.length; i++) {
      bias[i] -= lr * clipGrad(grad[i]);
    }
  }

  return loss;
}

function trainBatch(model, samples, lr) {
  const acc = createGradAccumulator();
  let totalLoss = 0;
  let count = 0;
  for (const { input, target } of samples) {
    const { probs, cache } = forward(model, input);
    const loss = crossEntropyLoss(probs, target);
    if (loss > 20 || isNaN(loss)) continue;
    const grads = backward(model, cache, target);
    accumulateGrads(acc, grads);
    totalLoss += loss;
    count++;
  }
  if (count > 0) {
    applyGrads(model, acc, lr, count);
  }
  return count > 0 ? totalLoss / count : NaN;
}

// --- Numerical Gradient (for verification) ---

function numericalGradient(model, input, target, paramName, i, j, epsilon) {
  epsilon = epsilon || 1e-5;
  const param = model[paramName];
  const isMatrix = Array.isArray(param) || (param[0] && param[0].length !== undefined);

  let original;
  if (isMatrix) {
    original = param[i][j];
    param[i][j] = original + epsilon;
  } else {
    original = param[i];
    param[i] = original + epsilon;
  }
  const { probs: probsPlus } = forward(model, input);
  const lossPlus = crossEntropyLoss(probsPlus, target);

  if (isMatrix) {
    param[i][j] = original - epsilon;
  } else {
    param[i] = original - epsilon;
  }
  const { probs: probsMinus } = forward(model, input);
  const lossMinus = crossEntropyLoss(probsMinus, target);

  // Restore
  if (isMatrix) {
    param[i][j] = original;
  } else {
    param[i] = original;
  }

  return (lossPlus - lossMinus) / (2 * epsilon);
}

// --- Exports ---

if (typeof module !== 'undefined') {
  module.exports = {
    NUM_TOKENS, EMBED_DIM, SEQ_LEN, FFN_DIM,
    MATRIX_PARAMS, BIAS_PARAMS,
    zeros, xavierInit, matVec, vecAdd, dot, softmax, gelu, geluDerivative,
    createModel, forward, crossEntropyLoss, backward,
    trainStep, trainBatch, createGradAccumulator, accumulateGrads, applyGrads,
    generateWaves, sampleTrainingExample, numericalGradient, clipGrad,
  };
}
