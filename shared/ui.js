// Tiny Transformer — Shared UI helpers (heatmap, texture rendering, sampling)

if (typeof require !== 'undefined' && typeof softmax === 'undefined') {
  var { softmax } = require('./core.js');
}

function heatColor(v) {
  var t = (Math.max(-1, Math.min(1, v)) + 1) / 2;
  var r = t < 0.5 ? 0 : (t - 0.5) * 2 * 255;
  var g = t < 0.25 ? t * 4 * 255 : t < 0.75 ? 255 : (1 - t) * 4 * 255;
  var b = t < 0.5 ? (0.5 - t) * 2 * 255 : 0;
  return [Math.round(Math.max(0, Math.min(255, r))), Math.round(Math.max(0, Math.min(255, g))), Math.round(Math.max(0, Math.min(255, b)))];
}

function renderTexture(canvas, data, rows, cols, trail, trailBuf) {
  if (!canvas) return;
  var cellSize = Math.max(2, Math.min(Math.floor(156 / cols), Math.floor(90 / rows), 10));
  var w = cols * cellSize, h = rows * cellSize;
  canvas.width = w;
  canvas.height = h;
  var ctx2d = canvas.getContext('2d');
  var img = ctx2d.createImageData(w, h);
  var min = Infinity, max = -Infinity;
  for (var i = 0; i < rows; i++) for (var j = 0; j < cols; j++) {
    var v = data[i][j];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  var range = max - min || 1;
  for (var i = 0; i < rows; i++) for (var j = 0; j < cols; j++) {
    var norm = ((data[i][j] - min) / range) * 2 - 1;
    if (trail && trailBuf) {
      var idx = i * cols + j;
      norm = trailBuf[idx] * 0.7 + norm * 0.3;
      trailBuf[idx] = norm;
    }
    var c = heatColor(norm);
    for (var dy = 0; dy < cellSize; dy++) for (var dx = 0; dx < cellSize; dx++) {
      var px = ((i * cellSize + dy) * w + j * cellSize + dx) * 4;
      img.data[px] = c[0]; img.data[px + 1] = c[1]; img.data[px + 2] = c[2]; img.data[px + 3] = 255;
    }
  }
  ctx2d.putImageData(img, 0, 0);
}

function sampleWithTemp(logits, temp) {
  if (temp <= 0.01) {
    var best = 0;
    for (var i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  var scaled = new Float64Array(logits.length);
  for (var i = 0; i < logits.length; i++) scaled[i] = logits[i] / temp;
  var probs = softmax(scaled);
  var r = Math.random(), cum = 0;
  for (var i = 0; i < probs.length; i++) { cum += probs[i]; if (r < cum) return i; }
  return probs.length - 1;
}

if (typeof module !== 'undefined') module.exports = { heatColor: heatColor, renderTexture: renderTexture, sampleWithTemp: sampleWithTemp };
