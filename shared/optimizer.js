// Tiny Transformer — Adam optimizer and training utilities

if (typeof require !== 'undefined' && typeof zeros === 'undefined') {
  var { zeros } = require('./core.js');
  var { MATRIX_PARAMS, BIAS_PARAMS, forward, backward } = require('./model.js');
}

function createAdam(model) {
  var state = {};
  for (var n of MATRIX_PARAMS) {
    state[n] = { m: zeros(model[n].length, model[n][0].length), v: zeros(model[n].length, model[n][0].length) };
  }
  for (var n of BIAS_PARAMS) {
    state[n] = { m: new Float64Array(model[n].length), v: new Float64Array(model[n].length) };
  }
  state.t = 0;
  return state;
}

function clipGrad(v) { return v > 5 ? 5 : v < -5 ? -5 : v; }

function adamStep(model, grads, adam, lr, beta1, beta2, eps) {
  beta1 = beta1 || 0.9; beta2 = beta2 || 0.999; eps = eps || 1e-8;
  adam.t++;
  var bc1 = 1 - Math.pow(beta1, adam.t);
  var bc2 = 1 - Math.pow(beta2, adam.t);
  for (var n of MATRIX_PARAMS) {
    var p = model[n], g = grads[n], am = adam[n].m, av = adam[n].v;
    for (var i = 0; i < p.length; i++)
      for (var j = 0; j < p[i].length; j++) {
        var gc = clipGrad(g[i][j]);
        am[i][j] = beta1 * am[i][j] + (1 - beta1) * gc;
        av[i][j] = beta2 * av[i][j] + (1 - beta2) * gc * gc;
        p[i][j] -= lr * (am[i][j] / bc1) / (Math.sqrt(av[i][j] / bc2) + eps);
      }
  }
  for (var n of BIAS_PARAMS) {
    var p = model[n], g = grads[n], am = adam[n].m, av = adam[n].v;
    for (var i = 0; i < p.length; i++) {
      var gc = clipGrad(g[i]);
      am[i] = beta1 * am[i] + (1 - beta1) * gc;
      av[i] = beta2 * av[i] + (1 - beta2) * gc * gc;
      p[i] -= lr * (am[i] / bc1) / (Math.sqrt(av[i] / bc2) + eps);
    }
  }
}

function trainStep(model, adam, input, target, lr) {
  var result = forward(model, input);
  var loss = -Math.log(Math.max(result.probs[target], 1e-12));
  if (loss > 20 || isNaN(loss)) return { loss: loss, probs: null, cache: null };
  var g = backward(model, result.cache, target);
  adamStep(model, g, adam, lr);
  return { loss: loss, probs: result.probs, cache: result.cache };
}

if (typeof module !== 'undefined') module.exports = { createAdam: createAdam, clipGrad: clipGrad, adamStep: adamStep, trainStep: trainStep };
