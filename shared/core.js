// Tiny Transformer — Core linear algebra and activations
// Shared across all demos. No model knowledge here.

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

function geluDerivative(x) {
  const c = Math.sqrt(2 / Math.PI);
  const z = c * (x + 0.044715 * x * x * x);
  const th = Math.tanh(z);
  return 0.5 * (1 + th) + 0.5 * x * (1 - th * th) * c * (1 + 3 * 0.044715 * x * x);
}

if (typeof module !== 'undefined') module.exports = { zeros, xavierInit, matVec, vecAdd, dot, softmax, gelu, geluDerivative };
