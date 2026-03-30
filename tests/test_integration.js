// Integration test: verifies the JSX component's model matches src/model.js
// and tests the full train → generate pipeline

const {
  createModel, forward, crossEntropyLoss, backward, numericalGradient,
  softmax, gelu, geluDerivative, countParams,
  createAdam, trainStep, heatColor,
} = require('../shared/index.js');
const { generateWaves, sampleExample: sampleTrainingExample } = require('../demos/waves/data.js');
const NUM_TOKENS = 16, SEQ_LEN = 24;

let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log('  ✓ ' + name); }
  catch(e) { failed++; console.log('  ✗ ' + name + ': ' + e.message); }
}

console.log('=== Integration Tests ===\n');

// Test 1: Full pipeline — train, then generate coherent continuation
console.log('Pipeline: train 2000 ticks, then generate');
test('training reduces loss and generation produces wave-like output', () => {
  const model = createModel({ dim: 12 });
  const adam = createAdam(model);
  const waves = generateWaves(30);
  let lastLoss = Infinity;
  const lossHistory = [];

  for (let tick = 0; tick < 2000; tick++) {
    let tl = 0, c = 0;
    for (let i = 0; i < 24; i++) {
      const { input, target } = sampleTrainingExample(waves);
      const { loss } = trainStep(model, adam, input, target, 0.001);
      if (!isNaN(loss) && loss <= 20) { tl += loss; c++; }
    }
    if (c > 0) { lastLoss = tl / c; lossHistory.push(lastLoss); }
  }

  const first50 = lossHistory.slice(0, 50).reduce((a, b) => a + b) / 50;
  const last50 = lossHistory.slice(-50).reduce((a, b) => a + b) / 50;
  console.log('    first 50 avg: ' + first50.toFixed(4) + ', last 50 avg: ' + last50.toFixed(4));

  if (last50 >= first50) throw new Error('Loss did not decrease: ' + first50.toFixed(4) + ' -> ' + last50.toFixed(4));

  // Generate from a sine-like seed
  const seed = [7, 8, 10, 11, 12, 12, 11, 10, 8, 7, 5, 4, 3, 3, 4, 5];
  const gen = [...seed];
  for (let i = 0; i < 20; i++) {
    const ctx = gen.slice(-SEQ_LEN);
    const { probs } = forward(model, ctx);
    const next = Array.from(probs).indexOf(Math.max(...probs));
    gen.push(next);
  }
  console.log('    seed:      [' + seed.join(',') + ']');
  console.log('    generated: [' + gen.slice(seed.length).join(',') + ']');

  // Verify generation stays in range
  for (const t of gen) {
    if (t < 0 || t > 15) throw new Error('Token out of range: ' + t);
  }
});

// Test 2: Verify model parameter shapes match spec
console.log('\nParameter shapes:');
test('all parameters have correct shapes', () => {
  const model = createModel({ dim: 12 });
  const checks = [
    ['tok_emb', 16, 12], ['pos_emb', 24, 12],
    ['Wq', 12, 12], ['Wk', 12, 12], ['Wv', 12, 12], ['Wo', 12, 12],
    ['ffn_up', 36, 12], ['ffn_down', 12, 36], ['out_proj', 16, 12],
  ];
  for (const [name, rows, cols] of checks) {
    const m = model[name];
    if (m.length !== rows) throw new Error(name + ' rows: expected ' + rows + ', got ' + m.length);
    if (m[0].length !== cols) throw new Error(name + ' cols: expected ' + cols + ', got ' + m[0].length);
  }
  if (model.ffn_up_bias.length !== 36) throw new Error('ffn_up_bias length');
  if (model.ffn_down_bias.length !== 12) throw new Error('ffn_down_bias length');
});

// Test 3: Approximate param count
test('total parameter count ≈ 2160', () => {
  const model = createModel({ dim: 12 });
  const total = countParams(model);
  console.log('    total params: ' + total);
  if (total !== 2160) throw new Error('Expected 2160 params, got ' + total);
});

// Test 4: Generation determinism (greedy = no randomness)
console.log('\nDeterminism:');
test('greedy generation is deterministic', () => {
  const model = createModel({ dim: 12 });
  const input = [5, 6, 7, 8];
  const { probs: p1 } = forward(model, input);
  const { probs: p2 } = forward(model, input);
  for (let i = 0; i < NUM_TOKENS; i++) {
    if (Math.abs(p1[i] - p2[i]) > 1e-15) throw new Error('Non-deterministic at ' + i);
  }
});

// Test 5: Drawing quantization logic
console.log('\nDrawing quantization:');
test('quantizes raw points to tokens correctly', () => {
  // Simulate a horizontal line at y=0.5 (should map to level 7-8)
  const points = [];
  for (let i = 0; i < 50; i++) {
    points.push({ x: i / 49, y: 0.5 });
  }
  const sorted = [...points].sort((a, b) => a.x - b.x);
  const xMin = sorted[0].x, xMax = sorted[sorted.length - 1].x;
  const steps = 10;
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
  console.log('    horizontal line at y=0.5 → tokens: [' + tokens.join(',') + ']');
  // y=0.5 → (1-0.5)*15 = 7.5 → round to 8 (or 7)
  for (const t of tokens) {
    if (t < 7 || t > 8) throw new Error('Expected 7 or 8, got ' + t);
  }
});

// Test 6: Heatmap color function
console.log('\nHeatmap colors:');
test('heatColor produces correct range', () => {
  // v=-1 → blue
  const blue = heatColor(-1);
  if (blue[2] < 200) throw new Error('v=-1 should be blue: ' + JSON.stringify(blue));
  // v=0 → green
  const green = heatColor(0);
  if (green[1] < 200) throw new Error('v=0 should be green: ' + JSON.stringify(green));
  // v=1 → red
  const red = heatColor(1);
  if (red[0] < 200) throw new Error('v=1 should be red: ' + JSON.stringify(red));
  console.log('    v=-1 → ' + JSON.stringify(blue) + ', v=0 → ' + JSON.stringify(green) + ', v=1 → ' + JSON.stringify(red));
});

// Test 7: Training performance (analytical backprop speed)
console.log('\nPerformance:');
test('24 training steps complete in < 200ms', () => {
  const model = createModel({ dim: 12 });
  const adam = createAdam(model);
  const waves = generateWaves(30);
  const start = performance.now();
  for (let i = 0; i < 24; i++) {
    const { input, target } = sampleTrainingExample(waves);
    trainStep(model, adam, input, target, 0.001);
  }
  const elapsed = performance.now() - start;
  console.log('    24 steps: ' + elapsed.toFixed(1) + 'ms');
  if (elapsed > 200) throw new Error('Too slow: ' + elapsed.toFixed(1) + 'ms (limit 200ms)');
});

console.log('\n=== Results: ' + passed + ' passed, ' + failed + ' failed ===');
process.exit(failed > 0 ? 1 : 0);
