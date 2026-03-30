// Tiny Transformer — Model creation, forward pass, backward pass
// Parameterized by config: { numTokens, seqLen, dim }

if (typeof require !== 'undefined' && typeof zeros === 'undefined') {
  var { zeros, xavierInit, matVec, vecAdd, dot, softmax, gelu, geluDerivative } = require('./core.js');
}

var MATRIX_PARAMS = ['tok_emb','pos_emb','Wq','Wk','Wv','Wo','ffn_up','ffn_down','out_proj'];
var BIAS_PARAMS = ['ffn_up_bias','ffn_down_bias'];

function createModel(config) {
  config = config || {};
  var numTokens = config.numTokens || 16;
  var seqLen = config.seqLen || 24;
  var dim = config.dim || 24;
  var ffnDim = dim * 3;
  return {
    numTokens: numTokens, seqLen: seqLen, dim: dim, ffnDim: ffnDim,
    tok_emb: xavierInit(numTokens, dim),
    pos_emb: (function() {
      var m = xavierInit(seqLen, dim);
      var base = Math.sqrt(2 / (seqLen + dim));
      for (var i = 0; i < seqLen; i++)
        for (var j = 0; j < dim; j++) m[i][j] *= 0.1 / base;
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
    out_proj: xavierInit(numTokens, dim),
  };
}

function countParams(model) {
  var n = 0;
  for (var k of MATRIX_PARAMS) n += model[k].length * model[k][0].length;
  for (var k of BIAS_PARAMS) n += model[k].length;
  return n;
}

function forward(model, tokens) {
  var numTokens = model.numTokens, dim = model.dim, ffnDim = model.ffnDim;
  var T = tokens.length;
  var sqrtD = Math.sqrt(dim);
  var emb = [];
  for (var t = 0; t < T; t++) emb.push(vecAdd(model.tok_emb[tokens[t]], model.pos_emb[t]));
  var Q = [], K = [], V = [];
  for (var t = 0; t < T; t++) {
    Q.push(matVec(model.Wq, emb[t]));
    K.push(matVec(model.Wk, emb[t]));
    V.push(matVec(model.Wv, emb[t]));
  }
  var attnW = [], ctx = [];
  for (var i = 0; i < T; i++) {
    var sc = new Float64Array(i + 1);
    for (var j = 0; j <= i; j++) sc[j] = dot(Q[i], K[j]) / sqrtD;
    var w = softmax(sc);
    attnW.push(w);
    var c = new Float64Array(dim);
    for (var j = 0; j <= i; j++) for (var d = 0; d < dim; d++) c[d] += w[j] * V[j][d];
    ctx.push(c);
  }
  var attnOut = [], res1 = [];
  for (var t = 0; t < T; t++) {
    var ao = matVec(model.Wo, ctx[t]);
    attnOut.push(ao);
    res1.push(vecAdd(emb[t], ao));
  }
  var ffnPre = [], ffnAct = [], res2 = [];
  for (var t = 0; t < T; t++) {
    var pre = vecAdd(matVec(model.ffn_up, res1[t]), model.ffn_up_bias);
    ffnPre.push(pre);
    var act = new Float64Array(ffnDim);
    for (var j = 0; j < ffnDim; j++) act[j] = gelu(pre[j]);
    ffnAct.push(act);
    var out = vecAdd(matVec(model.ffn_down, act), model.ffn_down_bias);
    res2.push(vecAdd(res1[t], out));
  }
  var logits = matVec(model.out_proj, res2[T - 1]);
  var probs = softmax(logits);
  return { probs: probs, logits: logits, cache: { tokens: tokens, T: T, emb: emb, Q: Q, K: K, V: V, attnW: attnW, ctx: ctx, attnOut: attnOut, res1: res1, ffnPre: ffnPre, ffnAct: ffnAct, res2: res2, logits: logits } };
}

function backward(model, cache, target) {
  var numTokens = model.numTokens, seqLen = model.seqLen, dim = model.dim, ffnDim = model.ffnDim;
  var tokens = cache.tokens, T = cache.T, emb = cache.emb, Q = cache.Q, K = cache.K, V = cache.V;
  var attnW = cache.attnW, ctx = cache.ctx, res1 = cache.res1, ffnPre = cache.ffnPre, ffnAct = cache.ffnAct, res2 = cache.res2, logits = cache.logits;
  var sqrtD = Math.sqrt(dim);
  var g = {
    tok_emb: zeros(numTokens, dim), pos_emb: zeros(seqLen, dim),
    Wq: zeros(dim, dim), Wk: zeros(dim, dim), Wv: zeros(dim, dim), Wo: zeros(dim, dim),
    ffn_up: zeros(ffnDim, dim), ffn_up_bias: new Float64Array(ffnDim),
    ffn_down: zeros(dim, ffnDim), ffn_down_bias: new Float64Array(dim),
    out_proj: zeros(numTokens, dim),
  };
  var probs = softmax(logits);
  var dL = new Float64Array(numTokens);
  for (var i = 0; i < numTokens; i++) dL[i] = probs[i] - (i === target ? 1 : 0);

  var dR2 = [];
  for (var t = 0; t < T; t++) dR2.push(new Float64Array(dim));
  for (var i = 0; i < numTokens; i++)
    for (var j = 0; j < dim; j++) g.out_proj[i][j] = dL[i] * res2[T-1][j];
  for (var j = 0; j < dim; j++)
    for (var i = 0; i < numTokens; i++) dR2[T-1][j] += model.out_proj[i][j] * dL[i];

  var dR1 = [];
  for (var t = 0; t < T; t++) dR1.push(new Float64Array(dim));
  for (var t = 0; t < T; t++) {
    var dFO = new Float64Array(dR2[t]);
    for (var j = 0; j < dim; j++) dR1[t][j] += dR2[t][j];
    for (var j = 0; j < dim; j++) g.ffn_down_bias[j] += dFO[j];
    var dFA = new Float64Array(ffnDim);
    for (var i = 0; i < dim; i++)
      for (var j = 0; j < ffnDim; j++) {
        g.ffn_down[i][j] += dFO[i] * ffnAct[t][j];
        dFA[j] += model.ffn_down[i][j] * dFO[i];
      }
    var dFP = new Float64Array(ffnDim);
    for (var j = 0; j < ffnDim; j++) dFP[j] = dFA[j] * geluDerivative(ffnPre[t][j]);
    for (var j = 0; j < ffnDim; j++) g.ffn_up_bias[j] += dFP[j];
    for (var i = 0; i < ffnDim; i++)
      for (var j = 0; j < dim; j++) {
        g.ffn_up[i][j] += dFP[i] * res1[t][j];
        dR1[t][j] += model.ffn_up[i][j] * dFP[i];
      }
  }

  var dEmb = [];
  for (var t = 0; t < T; t++) dEmb.push(new Float64Array(dim));
  var dAO = [];
  for (var t = 0; t < T; t++) {
    dAO.push(new Float64Array(dR1[t]));
    for (var j = 0; j < dim; j++) dEmb[t][j] += dR1[t][j];
  }
  var dCtx = [];
  for (var t = 0; t < T; t++) {
    dCtx.push(new Float64Array(dim));
    for (var i = 0; i < dim; i++)
      for (var j = 0; j < dim; j++) {
        g.Wo[i][j] += dAO[t][i] * ctx[t][j];
        dCtx[t][j] += model.Wo[i][j] * dAO[t][i];
      }
  }
  var dQ = [], dK = [], dV = [];
  for (var t = 0; t < T; t++) { dQ.push(new Float64Array(dim)); dK.push(new Float64Array(dim)); dV.push(new Float64Array(dim)); }
  for (var i = 0; i < T; i++) {
    var dW = new Float64Array(i + 1);
    for (var j = 0; j <= i; j++) { dW[j] = dot(dCtx[i], V[j]); for (var d = 0; d < dim; d++) dV[j][d] += attnW[i][j] * dCtx[i][d]; }
    var ws = 0; for (var j = 0; j <= i; j++) ws += attnW[i][j] * dW[j];
    for (var j = 0; j <= i; j++) { var ds = attnW[i][j] * (dW[j] - ws) / sqrtD; for (var d = 0; d < dim; d++) { dQ[i][d] += ds * K[j][d]; dK[j][d] += ds * Q[i][d]; } }
  }
  for (var t = 0; t < T; t++)
    for (var i = 0; i < dim; i++)
      for (var j = 0; j < dim; j++) {
        g.Wq[i][j] += dQ[t][i] * emb[t][j]; g.Wk[i][j] += dK[t][i] * emb[t][j]; g.Wv[i][j] += dV[t][i] * emb[t][j];
        dEmb[t][j] += model.Wq[i][j] * dQ[t][i] + model.Wk[i][j] * dK[t][i] + model.Wv[i][j] * dV[t][i];
      }
  for (var t = 0; t < T; t++)
    for (var d = 0; d < dim; d++) {
      g.tok_emb[tokens[t]][d] += dEmb[t][d];
      g.pos_emb[t][d] += dEmb[t][d];
    }
  return g;
}

function crossEntropyLoss(probs, target) {
  return -Math.log(Math.max(probs[target], 1e-12));
}

function numericalGradient(model, input, target, paramName, i, j, epsilon) {
  epsilon = epsilon || 1e-5;
  var param = model[paramName];
  var isMatrix = Array.isArray(param) || (param[0] && param[0].length !== undefined);
  var original;
  if (isMatrix) { original = param[i][j]; param[i][j] = original + epsilon; }
  else { original = param[i]; param[i] = original + epsilon; }
  var lossPlus = crossEntropyLoss(forward(model, input).probs, target);
  if (isMatrix) { param[i][j] = original - epsilon; } else { param[i] = original - epsilon; }
  var lossMinus = crossEntropyLoss(forward(model, input).probs, target);
  if (isMatrix) { param[i][j] = original; } else { param[i] = original; }
  return (lossPlus - lossMinus) / (2 * epsilon);
}

if (typeof module !== 'undefined') module.exports = { MATRIX_PARAMS: MATRIX_PARAMS, BIAS_PARAMS: BIAS_PARAMS, createModel: createModel, countParams: countParams, forward: forward, backward: backward, crossEntropyLoss: crossEntropyLoss, numericalGradient: numericalGradient };
