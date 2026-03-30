// Wave Demo — Data generation and sampling

function generateWaves(count) {
  count = count || 30;
  var waves = [];
  for (var i = 0; i < count; i++) {
    var freq = 0.04 + Math.random() * 0.3, amp = 4 + Math.random() * 3.5;
    var phase = Math.random() * 2 * Math.PI, offset = 5.5 + Math.random() * 4;
    var len = 60 + Math.floor(Math.random() * 41);
    var w = [];
    for (var t = 0; t < len; t++) w.push(Math.max(0, Math.min(15, Math.round(offset + amp * Math.sin(2 * Math.PI * freq * t + phase)))));
    waves.push(w);
  }
  return waves;
}

function sampleExample(waves, seqLen) {
  seqLen = seqLen || 24;
  var w = waves[Math.floor(Math.random() * waves.length)];
  var st = Math.floor(Math.random() * (w.length - 3));
  var wl = 3 + Math.floor(Math.random() * (Math.min(seqLen, w.length - st - 1) - 2));
  return { input: w.slice(st, st + wl), target: w[st + wl] };
}

if (typeof module !== 'undefined') module.exports = { generateWaves: generateWaves, sampleExample: sampleExample };
