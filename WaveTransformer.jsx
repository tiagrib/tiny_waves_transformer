import { useState, useRef, useEffect, useCallback, useMemo } from "react";

// === HYPERPARAMETERS ===
const NUM_TOKENS = 16;
const SEQ_LEN = 24;

// === LINEAR ALGEBRA ===
function zeros(r, c) {
  if (c === undefined) return new Float64Array(r);
  const m = [];
  for (let i = 0; i < r; i++) m.push(new Float64Array(c));
  return m;
}
function xavierInit(r, c) {
  const s = Math.sqrt(2 / (r + c));
  const m = [];
  for (let i = 0; i < r; i++) {
    const row = new Float64Array(c);
    for (let j = 0; j < c; j++) row[j] = (Math.random() * 2 - 1) * s;
    m.push(row);
  }
  return m;
}
function matVec(mat, vec) {
  const out = new Float64Array(mat.length);
  for (let i = 0; i < mat.length; i++) {
    let s = 0;
    for (let j = 0; j < vec.length; j++) s += mat[i][j] * vec[j];
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
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const exps = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) { exps[i] = Math.exp(logits[i] - max); sum += exps[i]; }
  for (let i = 0; i < logits.length; i++) exps[i] /= sum;
  return exps;
}
function gelu(x) {
  const c = Math.sqrt(2 / Math.PI);
  return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
}
function geluDeriv(x) {
  const c = Math.sqrt(2 / Math.PI);
  const z = c * (x + 0.044715 * x * x * x);
  const th = Math.tanh(z);
  return 0.5 * (1 + th) + 0.5 * x * (1 - th * th) * c * (1 + 3 * 0.044715 * x * x);
}

// === MODEL ===
function createModel(dim) {
  const ffnDim = dim * 3;
  return {
    dim, ffnDim,
    tok_emb: xavierInit(NUM_TOKENS, dim),
    pos_emb: (() => {
      const m = xavierInit(SEQ_LEN, dim);
      const base = Math.sqrt(2 / (SEQ_LEN + dim));
      for (let i = 0; i < SEQ_LEN; i++)
        for (let j = 0; j < dim; j++) m[i][j] *= 0.1 / base;
      return m;
    })(),
    Wq: xavierInit(dim, dim),
    Wk: xavierInit(dim, dim),
    Wv: xavierInit(dim, dim),
    Wo: xavierInit(dim, dim),
    ffn_up: xavierInit(ffnDim, dim),
    ffn_up_bias: new Float64Array(ffnDim),
    ffn_down: xavierInit(dim, ffnDim),
    ffn_down_bias: new Float64Array(dim),
    out_proj: xavierInit(NUM_TOKENS, dim),
  };
}

const MAT_PARAMS = ['tok_emb','pos_emb','Wq','Wk','Wv','Wo','ffn_up','ffn_down','out_proj'];
const BIAS_PARAMS = ['ffn_up_bias','ffn_down_bias'];

function countParams(model) {
  let n = 0;
  for (const k of MAT_PARAMS) n += model[k].length * model[k][0].length;
  for (const k of BIAS_PARAMS) n += model[k].length;
  return n;
}

function forward(model, tokens) {
  const { dim, ffnDim } = model;
  const T = tokens.length;
  const sqrtD = Math.sqrt(dim);
  const emb = [];
  for (let t = 0; t < T; t++) emb.push(vecAdd(model.tok_emb[tokens[t]], model.pos_emb[t]));
  const Q = [], K = [], V = [];
  for (let t = 0; t < T; t++) {
    Q.push(matVec(model.Wq, emb[t]));
    K.push(matVec(model.Wk, emb[t]));
    V.push(matVec(model.Wv, emb[t]));
  }
  const attnW = [], ctx = [];
  for (let i = 0; i < T; i++) {
    const sc = new Float64Array(i + 1);
    for (let j = 0; j <= i; j++) sc[j] = dot(Q[i], K[j]) / sqrtD;
    const w = softmax(sc);
    attnW.push(w);
    const c = new Float64Array(dim);
    for (let j = 0; j <= i; j++) for (let d = 0; d < dim; d++) c[d] += w[j] * V[j][d];
    ctx.push(c);
  }
  const attnOut = [], res1 = [];
  for (let t = 0; t < T; t++) {
    const ao = matVec(model.Wo, ctx[t]);
    attnOut.push(ao);
    res1.push(vecAdd(emb[t], ao));
  }
  const ffnPre = [], ffnAct = [], res2 = [];
  for (let t = 0; t < T; t++) {
    const pre = vecAdd(matVec(model.ffn_up, res1[t]), model.ffn_up_bias);
    ffnPre.push(pre);
    const act = new Float64Array(ffnDim);
    for (let j = 0; j < ffnDim; j++) act[j] = gelu(pre[j]);
    ffnAct.push(act);
    const out = vecAdd(matVec(model.ffn_down, act), model.ffn_down_bias);
    res2.push(vecAdd(res1[t], out));
  }
  const logits = matVec(model.out_proj, res2[T - 1]);
  const probs = softmax(logits);
  return { probs, logits, cache: { tokens, T, emb, Q, K, V, attnW, ctx, attnOut, res1, ffnPre, ffnAct, res2, logits } };
}

function backward(model, cache, target) {
  const { dim, ffnDim } = model;
  const { tokens, T, emb, Q, K, V, attnW, ctx, attnOut, res1, ffnPre, ffnAct, res2, logits } = cache;
  const sqrtD = Math.sqrt(dim);
  const g = {
    tok_emb: zeros(NUM_TOKENS, dim), pos_emb: zeros(SEQ_LEN, dim),
    Wq: zeros(dim, dim), Wk: zeros(dim, dim),
    Wv: zeros(dim, dim), Wo: zeros(dim, dim),
    ffn_up: zeros(ffnDim, dim), ffn_up_bias: new Float64Array(ffnDim),
    ffn_down: zeros(dim, ffnDim), ffn_down_bias: new Float64Array(dim),
    out_proj: zeros(NUM_TOKENS, dim),
  };
  const probs = softmax(logits);
  const dL = new Float64Array(NUM_TOKENS);
  for (let i = 0; i < NUM_TOKENS; i++) dL[i] = probs[i] - (i === target ? 1 : 0);

  const dR2 = [];
  for (let t = 0; t < T; t++) dR2.push(new Float64Array(dim));
  for (let i = 0; i < NUM_TOKENS; i++)
    for (let j = 0; j < dim; j++) g.out_proj[i][j] = dL[i] * res2[T-1][j];
  for (let j = 0; j < dim; j++)
    for (let i = 0; i < NUM_TOKENS; i++) dR2[T-1][j] += model.out_proj[i][j] * dL[i];

  const dR1 = [];
  for (let t = 0; t < T; t++) dR1.push(new Float64Array(dim));
  for (let t = 0; t < T; t++) {
    const dFO = new Float64Array(dR2[t]);
    for (let j = 0; j < dim; j++) dR1[t][j] += dR2[t][j];
    for (let j = 0; j < dim; j++) g.ffn_down_bias[j] += dFO[j];
    const dFA = new Float64Array(ffnDim);
    for (let i = 0; i < dim; i++)
      for (let j = 0; j < ffnDim; j++) {
        g.ffn_down[i][j] += dFO[i] * ffnAct[t][j];
        dFA[j] += model.ffn_down[i][j] * dFO[i];
      }
    const dFP = new Float64Array(ffnDim);
    for (let j = 0; j < ffnDim; j++) dFP[j] = dFA[j] * geluDeriv(ffnPre[t][j]);
    for (let j = 0; j < ffnDim; j++) g.ffn_up_bias[j] += dFP[j];
    for (let i = 0; i < ffnDim; i++)
      for (let j = 0; j < dim; j++) {
        g.ffn_up[i][j] += dFP[i] * res1[t][j];
        dR1[t][j] += model.ffn_up[i][j] * dFP[i];
      }
  }

  const dEmb = [];
  for (let t = 0; t < T; t++) dEmb.push(new Float64Array(dim));
  const dAO = [];
  for (let t = 0; t < T; t++) {
    dAO.push(new Float64Array(dR1[t]));
    for (let j = 0; j < dim; j++) dEmb[t][j] += dR1[t][j];
  }
  const dCtx = [];
  for (let t = 0; t < T; t++) {
    dCtx.push(new Float64Array(dim));
    for (let i = 0; i < dim; i++)
      for (let j = 0; j < dim; j++) {
        g.Wo[i][j] += dAO[t][i] * ctx[t][j];
        dCtx[t][j] += model.Wo[i][j] * dAO[t][i];
      }
  }
  const dQ = [], dK = [], dV = [];
  for (let t = 0; t < T; t++) { dQ.push(new Float64Array(dim)); dK.push(new Float64Array(dim)); dV.push(new Float64Array(dim)); }
  for (let i = 0; i < T; i++) {
    const dW = new Float64Array(i + 1);
    for (let j = 0; j <= i; j++) { dW[j] = dot(dCtx[i], V[j]); for (let d = 0; d < dim; d++) dV[j][d] += attnW[i][j] * dCtx[i][d]; }
    let ws = 0; for (let j = 0; j <= i; j++) ws += attnW[i][j] * dW[j];
    for (let j = 0; j <= i; j++) { const ds = attnW[i][j] * (dW[j] - ws) / sqrtD; for (let d = 0; d < dim; d++) { dQ[i][d] += ds * K[j][d]; dK[j][d] += ds * Q[i][d]; } }
  }
  for (let t = 0; t < T; t++)
    for (let i = 0; i < dim; i++)
      for (let j = 0; j < dim; j++) {
        g.Wq[i][j] += dQ[t][i] * emb[t][j]; g.Wk[i][j] += dK[t][i] * emb[t][j]; g.Wv[i][j] += dV[t][i] * emb[t][j];
        dEmb[t][j] += model.Wq[i][j] * dQ[t][i] + model.Wk[i][j] * dK[t][i] + model.Wv[i][j] * dV[t][i];
      }
  for (let t = 0; t < T; t++)
    for (let d = 0; d < dim; d++) {
      g.tok_emb[tokens[t]][d] += dEmb[t][d];
      g.pos_emb[t][d] += dEmb[t][d];
    }
  return g;
}

// === ADAM OPTIMIZER ===
function createAdam(model) {
  const state = {};
  for (const n of MAT_PARAMS) {
    state[n] = { m: zeros(model[n].length, model[n][0].length), v: zeros(model[n].length, model[n][0].length) };
  }
  for (const n of BIAS_PARAMS) {
    state[n] = { m: new Float64Array(model[n].length), v: new Float64Array(model[n].length) };
  }
  state.t = 0;
  return state;
}

function clip(v) { return v > 5 ? 5 : v < -5 ? -5 : v; }

function adamStep(model, grads, adam, lr, beta1, beta2, eps) {
  beta1 = beta1 || 0.9; beta2 = beta2 || 0.999; eps = eps || 1e-8;
  adam.t++;
  const bc1 = 1 - Math.pow(beta1, adam.t);
  const bc2 = 1 - Math.pow(beta2, adam.t);
  for (const n of MAT_PARAMS) {
    const p = model[n], g = grads[n], am = adam[n].m, av = adam[n].v;
    for (let i = 0; i < p.length; i++)
      for (let j = 0; j < p[i].length; j++) {
        const gc = clip(g[i][j]);
        am[i][j] = beta1 * am[i][j] + (1 - beta1) * gc;
        av[i][j] = beta2 * av[i][j] + (1 - beta2) * gc * gc;
        p[i][j] -= lr * (am[i][j] / bc1) / (Math.sqrt(av[i][j] / bc2) + eps);
      }
  }
  for (const n of BIAS_PARAMS) {
    const p = model[n], g = grads[n], am = adam[n].m, av = adam[n].v;
    for (let i = 0; i < p.length; i++) {
      const gc = clip(g[i]);
      am[i] = beta1 * am[i] + (1 - beta1) * gc;
      av[i] = beta2 * av[i] + (1 - beta2) * gc * gc;
      p[i] -= lr * (am[i] / bc1) / (Math.sqrt(av[i] / bc2) + eps);
    }
  }
}

function trainStep(model, adam, input, target, lr) {
  const { probs, cache } = forward(model, input);
  const loss = -Math.log(Math.max(probs[target], 1e-12));
  if (loss > 20 || isNaN(loss)) return { loss, probs: null, cache: null };
  const g = backward(model, cache, target);
  adamStep(model, g, adam, lr);
  return { loss, probs, cache };
}

// === DATA GENERATION ===
function generateWaves(count = 30) {
  const waves = [];
  for (let i = 0; i < count; i++) {
    const freq = 0.04 + Math.random() * 0.3, amp = 4 + Math.random() * 3.5;
    const phase = Math.random() * 2 * Math.PI, offset = 5.5 + Math.random() * 4;
    const len = 60 + Math.floor(Math.random() * 41);
    const w = [];
    for (let t = 0; t < len; t++) w.push(Math.max(0, Math.min(15, Math.round(offset + amp * Math.sin(2 * Math.PI * freq * t + phase)))));
    waves.push(w);
  }
  return waves;
}
function sampleExample(waves) {
  const w = waves[Math.floor(Math.random() * waves.length)];
  const st = Math.floor(Math.random() * (w.length - 3));
  const wl = 3 + Math.floor(Math.random() * (Math.min(SEQ_LEN, w.length - st - 1) - 2));
  return { input: w.slice(st, st + wl), target: w[st + wl] };
}

// === HEATMAP COLOR ===
function heatColor(v) {
  const t = (Math.max(-1, Math.min(1, v)) + 1) / 2;
  const r = t < 0.5 ? 0 : (t - 0.5) * 2 * 255;
  const g = t < 0.25 ? t * 4 * 255 : t < 0.75 ? 255 : (1 - t) * 4 * 255;
  const b = t < 0.5 ? (0.5 - t) * 2 * 255 : 0;
  return [Math.round(Math.max(0, Math.min(255, r))), Math.round(Math.max(0, Math.min(255, g))), Math.round(Math.max(0, Math.min(255, b)))];
}

// === TEXTURE RENDERING ===
function renderTexture(canvas, data, rows, cols, trail, trailBuf) {
  if (!canvas) return;
  const cellSize = Math.max(2, Math.min(Math.floor(156 / cols), Math.floor(90 / rows), 10));
  const w = cols * cellSize, h = rows * cellSize;
  canvas.width = w;
  canvas.height = h;
  const ctx2d = canvas.getContext('2d');
  const img = ctx2d.createImageData(w, h);
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const v = data[i][j];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    let norm = ((data[i][j] - min) / range) * 2 - 1;
    if (trail && trailBuf) {
      const idx = i * cols + j;
      norm = trailBuf[idx] * 0.7 + norm * 0.3;
      trailBuf[idx] = norm;
    }
    const [r, g, b] = heatColor(norm);
    for (let dy = 0; dy < cellSize; dy++) for (let dx = 0; dx < cellSize; dx++) {
      const px = ((i * cellSize + dy) * w + j * cellSize + dx) * 4;
      img.data[px] = r; img.data[px + 1] = g; img.data[px + 2] = b; img.data[px + 3] = 255;
    }
  }
  ctx2d.putImageData(img, 0, 0);
}

// === TEMPERATURE SAMPLING ===
function sampleWithTemp(logits, temp) {
  if (temp <= 0.01) {
    // Greedy
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  const scaled = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / temp;
  const probs = softmax(scaled);
  let r = Math.random(), cum = 0;
  for (let i = 0; i < probs.length; i++) { cum += probs[i]; if (r < cum) return i; }
  return probs.length - 1;
}

// === MAIN COMPONENT ===
export default function WaveTransformer() {
  const modelRef = useRef(null);
  const adamRef = useRef(null);
  const wavesRef = useRef(null);
  const epochRef = useRef(0);
  const trainIntervalRef = useRef(null);
  const genIntervalRef = useRef(null);
  const trailBufsRef = useRef({});
  const drawCanvasRef = useRef(null);
  const textureCanvasRefs = useRef({});
  const lastAttnRef = useRef(null);
  const lastFfnActRef = useRef(null);
  const lastProbsRef = useRef(null);

  const [phase, setPhase] = useState('DRAW'); // DRAW | IDLE | PLAYING | PAUSED
  const [isTraining, setIsTraining] = useState(false);
  const [seed, setSeed] = useState([]);
  const [generated, setGenerated] = useState([]);
  const [seedSteps, setSeedSteps] = useState(24);
  const [loss, setLoss] = useState(null);
  const [epoch, setEpoch] = useState(0);
  const [trail, setTrail] = useState(false);
  const [drawPoints, setDrawPoints] = useState([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [renderTick, setRenderTick] = useState(0);
  const lossHistoryRef = useRef([]);
  const [lr, setLr] = useState(0.001);
  const [dim, setDim] = useState(24);
  const [temp, setTemp] = useState(0.0);
  const [paramCount, setParamCount] = useState(0);

  // Init model
  useEffect(() => {
    const m = createModel(dim);
    modelRef.current = m;
    adamRef.current = createAdam(m);
    wavesRef.current = generateWaves(30);
    setParamCount(countParams(m));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Drawing handlers
  const canvasToNorm = useCallback((canvas, e) => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
  }, []);

  const quantizeDraw = useCallback((points, steps) => {
    if (points.length < 2) return [];
    const sorted = [...points].sort((a, b) => a.x - b.x);
    const xMin = sorted[0].x, xMax = sorted[sorted.length - 1].x;
    if (xMax - xMin < 0.01) return [];
    const tokens = [];
    for (let s = 0; s < steps; s++) {
      const tx = xMin + (s / (steps - 1)) * (xMax - xMin);
      let best = sorted[0], bestDist = Math.abs(sorted[0].x - tx);
      for (const p of sorted) {
        const d = Math.abs(p.x - tx);
        if (d < bestDist) { best = p; bestDist = d; }
      }
      tokens.push(Math.max(0, Math.min(15, Math.round((1 - best.y) * 15))));
    }
    return tokens;
  }, []);

  const handlePointerDown = useCallback((e) => {
    if (phase !== 'DRAW') return;
    const canvas = drawCanvasRef.current;
    if (!canvas) return;
    canvas.setPointerCapture(e.pointerId);
    setIsDrawing(true);
    const p = canvasToNorm(canvas, e);
    setDrawPoints([p]);
  }, [phase, canvasToNorm]);

  const handlePointerMove = useCallback((e) => {
    if (!isDrawing || phase !== 'DRAW') return;
    const canvas = drawCanvasRef.current;
    if (!canvas) return;
    const p = canvasToNorm(canvas, e);
    setDrawPoints(prev => [...prev, p]);
  }, [isDrawing, phase, canvasToNorm]);

  const handlePointerUp = useCallback(() => {
    if (!isDrawing) return;
    setIsDrawing(false);
    setDrawPoints(prev => {
      const tokens = quantizeDraw(prev, seedSteps);
      if (tokens.length >= 3) {
        setSeed(tokens);
        setGenerated([]);
        setPhase('IDLE');
      }
      return prev;
    });
  }, [isDrawing, quantizeDraw, seedSteps]);

  // Draw the wave canvas
  const drawWaveCanvas = useCallback(() => {
    const canvas = drawCanvasRef.current;
    if (!canvas) return;
    const ctx2d = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx2d.fillStyle = '#111';
    ctx2d.fillRect(0, 0, w, h);

    // Grid lines
    for (let level = 0; level <= 15; level++) {
      const y = h - (level / 15) * h;
      ctx2d.strokeStyle = level % 4 === 0 ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.07)';
      ctx2d.lineWidth = 1;
      ctx2d.beginPath(); ctx2d.moveTo(0, y); ctx2d.lineTo(w, y); ctx2d.stroke();
    }

    if (phase === 'DRAW') {
      const history = lossHistoryRef.current;
      if (drawPoints.length > 1) {
        // Raw stroke
        ctx2d.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx2d.lineWidth = 2;
        ctx2d.beginPath();
        ctx2d.moveTo(drawPoints[0].x * w, drawPoints[0].y * h);
        for (let i = 1; i < drawPoints.length; i++) ctx2d.lineTo(drawPoints[i].x * w, drawPoints[i].y * h);
        ctx2d.stroke();

        // Quantized preview
        const preview = quantizeDraw(drawPoints, seedSteps);
        if (preview.length > 0) {
          ctx2d.strokeStyle = 'white';
          ctx2d.lineWidth = 2;
          ctx2d.beginPath();
          for (let i = 0; i < preview.length; i++) {
            const x = (i / (preview.length - 1)) * w;
            const y = h - (preview[i] / 15) * h;
            if (i === 0) ctx2d.moveTo(x, y); else ctx2d.lineTo(x, y);
          }
          ctx2d.stroke();
          ctx2d.fillStyle = 'white';
          for (let i = 0; i < preview.length; i++) {
            const x = (i / (preview.length - 1)) * w;
            const y = h - (preview[i] / 15) * h;
            ctx2d.beginPath(); ctx2d.arc(x, y, 3, 0, 2 * Math.PI); ctx2d.fill();
          }
        }
      } else if (history.length > 1) {
        // Loss graph
        const pad = { top: 16, bottom: 20, left: 44, right: 12 };
        const gw = w - pad.left - pad.right, gh = h - pad.top - pad.bottom;
        const maxLoss = Math.max(...history.slice(0, 20)); // scale from early peak
        const minLoss = Math.min(...history);
        const yRange = Math.max(maxLoss - minLoss, 0.01);

        // Y-axis labels
        ctx2d.fillStyle = '#666';
        ctx2d.font = '10px monospace';
        ctx2d.textAlign = 'right';
        for (let i = 0; i <= 4; i++) {
          const v = minLoss + (yRange * (4 - i)) / 4;
          const y = pad.top + (i / 4) * gh;
          ctx2d.fillText(v.toFixed(2), pad.left - 4, y + 3);
          ctx2d.strokeStyle = 'rgba(255,255,255,0.06)';
          ctx2d.lineWidth = 1;
          ctx2d.beginPath(); ctx2d.moveTo(pad.left, y); ctx2d.lineTo(w - pad.right, y); ctx2d.stroke();
        }

        // X-axis label
        ctx2d.fillStyle = '#666';
        ctx2d.textAlign = 'center';
        ctx2d.fillText('epoch ' + history.length, w / 2, h - 2);

        // Loss curve
        ctx2d.strokeStyle = '#f84';
        ctx2d.lineWidth = 1.5;
        ctx2d.beginPath();
        for (let i = 0; i < history.length; i++) {
          const x = pad.left + (i / (history.length - 1)) * gw;
          const y = pad.top + ((maxLoss - history[i]) / yRange) * gh;
          if (i === 0) ctx2d.moveTo(x, y); else ctx2d.lineTo(x, y);
        }
        ctx2d.stroke();

        // Current loss label
        ctx2d.fillStyle = '#f84';
        ctx2d.textAlign = 'left';
        ctx2d.font = '11px monospace';
        ctx2d.fillText('loss ' + history[history.length - 1].toFixed(4), pad.left + 4, pad.top - 4);

        // Prompt to draw
        if (!isTraining) {
          ctx2d.fillStyle = 'rgba(255,255,255,0.4)';
          ctx2d.textAlign = 'center';
          ctx2d.font = '13px monospace';
          ctx2d.fillText('draw a seed wave ↑', w / 2, h / 2);
        }
      } else {
        // No history, no drawing — prompt
        ctx2d.fillStyle = 'rgba(255,255,255,0.3)';
        ctx2d.textAlign = 'center';
        ctx2d.font = '13px monospace';
        ctx2d.fillText(isTraining ? 'training...' : 'draw a seed wave', w / 2, h / 2);
      }
    } else {
      // Show seed + generated
      const full = [...seed, ...generated];
      if (full.length === 0) return;
      const total = full.length;
      const xScale = w / Math.max(total - 1, 1);

      // Seed portion
      ctx2d.strokeStyle = 'white';
      ctx2d.lineWidth = 2;
      ctx2d.beginPath();
      for (let i = 0; i < seed.length; i++) {
        const x = i * xScale, y = h - (seed[i] / 15) * h;
        if (i === 0) ctx2d.moveTo(x, y); else ctx2d.lineTo(x, y);
      }
      ctx2d.stroke();
      ctx2d.fillStyle = 'white';
      for (let i = 0; i < seed.length; i++) {
        const x = i * xScale, y = h - (seed[i] / 15) * h;
        ctx2d.beginPath(); ctx2d.arc(x, y, 3, 0, 2 * Math.PI); ctx2d.fill();
      }

      // Boundary line
      if (generated.length > 0) {
        const bx = (seed.length - 0.5) * xScale;
        ctx2d.setLineDash([4, 4]);
        ctx2d.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx2d.beginPath(); ctx2d.moveTo(bx, 0); ctx2d.lineTo(bx, h); ctx2d.stroke();
        ctx2d.setLineDash([]);

        // Generated portion
        ctx2d.strokeStyle = '#4f4';
        ctx2d.lineWidth = 2;
        ctx2d.beginPath();
        ctx2d.moveTo((seed.length - 1) * xScale, h - (seed[seed.length - 1] / 15) * h);
        for (let i = 0; i < generated.length; i++) {
          const x = (seed.length + i) * xScale, y = h - (generated[i] / 15) * h;
          ctx2d.lineTo(x, y);
        }
        ctx2d.stroke();
        ctx2d.fillStyle = '#4f4';
        for (let i = 0; i < generated.length; i++) {
          const x = (seed.length + i) * xScale, y = h - (generated[i] / 15) * h;
          ctx2d.beginPath(); ctx2d.arc(x, y, 3, 0, 2 * Math.PI); ctx2d.fill();
        }
      }
    }
  }, [phase, seed, generated, drawPoints, seedSteps, quantizeDraw, isTraining, epoch]);

  // Canvas resize
  useEffect(() => {
    const canvas = drawCanvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const parent = canvas.parentElement;
      if (parent) { canvas.width = parent.clientWidth; canvas.height = parent.clientHeight; }
      drawWaveCanvas();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas.parentElement);
    return () => ro.disconnect();
  }, [drawWaveCanvas]);

  // Training
  const startTraining = useCallback(() => {
    if (trainIntervalRef.current) return;
    setIsTraining(true);
    trainIntervalRef.current = setInterval(() => {
      const model = modelRef.current;
      const adam = adamRef.current;
      const waves = wavesRef.current;
      if (!model || !adam || !waves) return;
      let totalLoss = 0, count = 0;
      for (let i = 0; i < 24; i++) {
        const { input, target } = sampleExample(waves);
        const { loss: l } = trainStep(model, adam, input, target, lrRef.current);
        if (!isNaN(l) && l <= 20) { totalLoss += l; count++; }
      }
      epochRef.current++;
      const avgLoss = count > 0 ? totalLoss / count : null;
      if (avgLoss !== null) lossHistoryRef.current.push(avgLoss);
      setLoss(avgLoss);
      setEpoch(epochRef.current);
      setRenderTick(n => n + 1);
    }, 50);
  }, []);

  const stopTraining = useCallback(() => {
    if (trainIntervalRef.current) {
      clearInterval(trainIntervalRef.current);
      trainIntervalRef.current = null;
    }
    setIsTraining(false);
  }, []);

  // Generation
  const lrRef = useRef(lr);
  lrRef.current = lr;
  const tempRef = useRef(temp);
  tempRef.current = temp;
  const seedRef = useRef(seed);
  seedRef.current = seed;

  const generateOne = useCallback(() => {
    const model = modelRef.current;
    if (!model) return;
    const currentSeed = seedRef.current;
    setGenerated(prev => {
      const full = [...currentSeed, ...prev];
      const context = full.slice(-SEQ_LEN);
      const { probs, logits, cache } = forward(model, context);
      const next = sampleWithTemp(logits, tempRef.current);
      lastAttnRef.current = cache.attnW;
      lastFfnActRef.current = cache.ffnAct;
      lastProbsRef.current = probs;
      return [...prev, next];
    });
    setRenderTick(n => n + 1);
  }, []);

  const startPlaying = useCallback(() => {
    if (genIntervalRef.current) return;
    setPhase('PLAYING');
    genIntervalRef.current = setInterval(() => generateOne(), 150);
  }, [generateOne]);

  const stopPlaying = useCallback(() => {
    if (genIntervalRef.current) {
      clearInterval(genIntervalRef.current);
      genIntervalRef.current = null;
    }
  }, []);

  const handlePlay = useCallback(() => {
    if (phase === 'PLAYING') {
      stopPlaying();
      setPhase('PAUSED');
    } else {
      startPlaying();
    }
  }, [phase, startPlaying, stopPlaying]);

  const handleStepBack = useCallback(() => {
    if (phase !== 'PAUSED' || generated.length === 0) return;
    setGenerated(prev => prev.slice(0, -1));
  }, [phase, generated.length]);

  const handleStepForward = useCallback(() => {
    if (phase !== 'PAUSED') return;
    generateOne();
  }, [phase, generateOne]);

  const handleRewind = useCallback(() => {
    stopPlaying();
    setGenerated([]);
    lastAttnRef.current = null;
    lastFfnActRef.current = null;
    lastProbsRef.current = null;
    setPhase('IDLE');
  }, [stopPlaying]);

  const handleClear = useCallback(() => {
    stopPlaying();
    setSeed([]);
    setGenerated([]);
    setDrawPoints([]);
    lastAttnRef.current = null;
    lastFfnActRef.current = null;
    lastProbsRef.current = null;
    setPhase('DRAW');
  }, [stopPlaying]);

  const handleReset = useCallback(() => {
    stopPlaying();
    stopTraining();
    const m = createModel(dim);
    modelRef.current = m;
    adamRef.current = createAdam(m);
    wavesRef.current = generateWaves(30);
    epochRef.current = 0;
    trailBufsRef.current = {};
    lossHistoryRef.current = [];
    setSeed([]);
    setGenerated([]);
    setDrawPoints([]);
    setLoss(null);
    setEpoch(0);
    setParamCount(countParams(m));
    lastAttnRef.current = null;
    lastFfnActRef.current = null;
    lastProbsRef.current = null;
    setPhase('DRAW');
  }, [stopPlaying, stopTraining, dim]);

  // Cleanup
  useEffect(() => () => { stopPlaying(); stopTraining(); }, [stopPlaying, stopTraining]);

  // Texture data
  const textures = useMemo(() => {
    const m = modelRef.current;
    if (!m) return [];
    const d = m.dim, fd = m.ffnDim;
    const list = [
      { name: 'tok embed', data: m.tok_emb, rows: 16, cols: d },
      { name: 'pos embed', data: m.pos_emb, rows: 24, cols: d },
      { name: 'W_query', data: m.Wq, rows: d, cols: d },
      { name: 'W_key', data: m.Wk, rows: d, cols: d },
      { name: 'W_value', data: m.Wv, rows: d, cols: d },
      { name: 'W_out', data: m.Wo, rows: d, cols: d },
      { name: 'FFN ↑', data: m.ffn_up, rows: fd, cols: d },
      { name: 'FFN ↓', data: m.ffn_down, rows: d, cols: fd },
      { name: 'readout', data: m.out_proj, rows: 16, cols: d },
    ];
    if (lastAttnRef.current && generated.length > 0) {
      const aw = lastAttnRef.current;
      const T = aw.length;
      const padded = [];
      for (let i = 0; i < T; i++) {
        const row = new Float64Array(T);
        for (let j = 0; j <= i; j++) row[j] = aw[i][j];
        padded.push(row);
      }
      list.push({ name: 'attention', data: padded, rows: T, cols: T });
    }
    if (lastFfnActRef.current && generated.length > 0) {
      const fd2 = modelRef.current ? modelRef.current.ffnDim : 36;
      list.push({ name: 'FFN act', data: lastFfnActRef.current, rows: lastFfnActRef.current.length, cols: fd2 });
    }
    return list;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [epoch, generated.length, renderTick]);

  // Render textures
  useEffect(() => {
    for (const tex of textures) {
      const canvas = textureCanvasRefs.current[tex.name];
      if (!canvas) continue;
      if (!trailBufsRef.current[tex.name]) trailBufsRef.current[tex.name] = new Float64Array(tex.rows * tex.cols);
      const tb = trail ? trailBufsRef.current[tex.name] : null;
      renderTexture(canvas, tex.data, tex.rows, tex.cols, trail, tb);
    }
  }, [textures, trail]);

  // Probability bar
  const probBar = lastProbsRef.current && generated.length > 0 ? lastProbsRef.current : null;

  const btnStyle = (active, disabled) => ({
    padding: '6px 10px', border: 'none', borderRadius: 4, cursor: disabled ? 'default' : 'pointer',
    fontSize: 13, fontFamily: 'monospace', fontWeight: 600,
    background: active ? '#c22' : disabled ? '#333' : '#444',
    color: disabled ? '#666' : '#eee',
    opacity: disabled ? 0.5 : 1,
  });

  return (
    <div style={{ background: '#0a0a0a', color: '#eee', fontFamily: 'monospace', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{ padding: '8px 12px', borderBottom: '1px solid #333', fontSize: 13, color: '#888' }}>
        <span style={{ color: '#eee', fontWeight: 700 }}>wave.transformer</span>
        {' '}{paramCount.toLocaleString()} params · 16 levels · {dim}d
      </div>

      {/* Toolbar */}
      <div style={{ padding: '6px 12px', display: 'flex', gap: 6, flexWrap: 'wrap', borderBottom: '1px solid #333', alignItems: 'center' }}>
        <button style={btnStyle(isTraining)} onClick={isTraining ? stopTraining : startTraining}>
          {isTraining ? '■ Stop' : '⟳ Train'}
        </button>
        <button style={btnStyle(phase === 'PLAYING', phase === 'DRAW')} onClick={handlePlay} disabled={phase === 'DRAW'}>
          {phase === 'PLAYING' ? '❚❚ Pause' : '▶ Play'}
        </button>
        <button style={btnStyle(false, phase !== 'PAUSED' || generated.length === 0)} onClick={handleStepBack} disabled={phase !== 'PAUSED' || generated.length === 0}>◀</button>
        <button style={btnStyle(false, phase !== 'PAUSED')} onClick={handleStepForward} disabled={phase !== 'PAUSED'}>▶</button>
        <button style={btnStyle(false, phase === 'DRAW')} onClick={handleRewind} disabled={phase === 'DRAW'}>↺ Rewind</button>
        <button style={btnStyle()} onClick={handleClear}>✕ Clear</button>
        <div style={{ flex: 1 }} />
        <button style={{ ...btnStyle(), background: '#622' }} onClick={handleReset}>RESET</button>
      </div>

      {/* Status bar */}
      <div style={{ padding: '6px 12px', display: 'flex', gap: 10, alignItems: 'center', borderBottom: '1px solid #333', fontSize: 11, flexWrap: 'wrap' }}>
        <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', border: '2px solid #888', background: trail ? '#4f4' : 'transparent', display: 'inline-block' }}
            onClick={() => setTrail(t => !t)} />
          TRAIL
        </label>
        <span>seed</span>
        <input type="range" min={8} max={48} value={seedSteps} onChange={e => setSeedSteps(+e.target.value)}
          disabled={phase !== 'DRAW'} style={{ width: 60 }} />
        <span>{seedSteps}</span>
        <span style={{ color: '#555' }}>|</span>
        <span>dim</span>
        <input type="range" min={8} max={48} step={4} value={dim}
          onChange={e => setDim(+e.target.value)}
          disabled={isTraining || phase !== 'DRAW'}
          style={{ width: 50 }} />
        <span>{dim}</span>
        <span style={{ color: '#555' }}>|</span>
        <span>lr</span>
        <input type="range" min={-4} max={-1} step={0.1} value={Math.log10(lr)}
          onChange={e => setLr(Math.round(10 ** +e.target.value * 10000) / 10000)}
          style={{ width: 50 }} />
        <span>{lr}</span>
        <span style={{ color: '#555' }}>|</span>
        <span>temp</span>
        <input type="range" min={0} max={1.5} step={0.05} value={temp}
          onChange={e => setTemp(+e.target.value)}
          style={{ width: 50 }} />
        <span>{temp.toFixed(2)}</span>
        <span style={{ color: '#555' }}>|</span>
        <span>ep {epoch}</span>
        <span>loss {loss !== null ? loss.toFixed(4) : '—'}</span>
        <span>{generated.length} gen</span>
      </div>

      {/* Wave Canvas */}
      <div style={{ height: '25vh', minHeight: 120, background: '#111', position: 'relative' }}>
        <canvas
          ref={drawCanvasRef}
          style={{ width: '100%', height: '100%', touchAction: 'none', cursor: phase === 'DRAW' ? 'crosshair' : 'default' }}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={handlePointerUp}
        />
      </div>

      {/* Texture Panel */}
      <div style={{ flex: 1, overflow: 'auto', padding: 8 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 8 }}>
          {textures.map(tex => (
            <div key={tex.name} style={{ textAlign: 'center' }}>
              <canvas
                ref={el => { if (el) textureCanvasRefs.current[tex.name] = el; }}
                style={{ imageRendering: 'pixelated', maxWidth: '100%' }}
              />
              <div style={{ fontSize: 10, color: '#888', marginTop: 2 }}>
                {tex.name} {tex.rows}×{tex.cols}
              </div>
            </div>
          ))}
        </div>

        {/* Probability bar */}
        {probBar && (
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 10, color: '#888', marginBottom: 4 }}>output probabilities</div>
            <div style={{ display: 'flex', gap: 2, alignItems: 'end', height: 60 }}>
              {Array.from(probBar).map((p, i) => (
                <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <div style={{ width: '100%', background: '#4f4', height: Math.max(1, p * 60), borderRadius: 2 }} />
                  <div style={{ fontSize: 8, color: '#666', marginTop: 2 }}>{i}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
