const assert = require('assert');
const {
  NUM_TOKENS, EMBED_DIM, SEQ_LEN, FFN_DIM,
  softmax, gelu, geluDerivative,
  createModel, forward, crossEntropyLoss, backward,
  generateWaves, sampleTrainingExample, numericalGradient,
} = require('../src/model.js');

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  ✓ ${name}`);
  } catch (e) {
    failed++;
    console.log(`  ✗ ${name}: ${e.message}`);
  }
}

function approxEqual(a, b, tol) {
  tol = tol || 1e-6;
  return Math.abs(a - b) < tol;
}

console.log('=== Model Tests ===\n');

// --- Softmax ---
console.log('Softmax:');
test('sums to 1', () => {
  const p = softmax([1, 2, 3, 4]);
  const sum = p.reduce((a, b) => a + b, 0);
  assert(approxEqual(sum, 1.0, 1e-10), `sum=${sum}`);
});

test('handles large values', () => {
  const p = softmax([1000, 1001, 999]);
  const sum = p.reduce((a, b) => a + b, 0);
  assert(approxEqual(sum, 1.0, 1e-10), `sum=${sum}`);
  assert(!isNaN(p[0]) && !isNaN(p[1]) && !isNaN(p[2]), 'NaN in output');
});

test('uniform for equal inputs', () => {
  const p = softmax([5, 5, 5, 5]);
  for (const v of p) assert(approxEqual(v, 0.25, 1e-6), `v=${v}`);
});

// --- GELU ---
console.log('\nGELU:');
test('gelu(0) ≈ 0', () => {
  assert(approxEqual(gelu(0), 0, 1e-6), `gelu(0)=${gelu(0)}`);
});

test('gelu(3) ≈ 3', () => {
  assert(approxEqual(gelu(3), 2.9960, 0.01), `gelu(3)=${gelu(3)}`);
});

test('gelu(-3) ≈ -0.004', () => {
  assert(approxEqual(gelu(-3), -0.00404, 0.001), `gelu(-3)=${gelu(-3)}`);
});

// --- GELU Derivative (numerical check) ---
console.log('\nGELU Derivative:');
test('matches numerical derivative', () => {
  const eps = 1e-5;
  for (const x of [-2, -1, 0, 0.5, 1, 2, 3]) {
    const analytical = geluDerivative(x);
    const numerical = (gelu(x + eps) - gelu(x - eps)) / (2 * eps);
    const relErr = Math.abs(analytical - numerical) / Math.max(Math.abs(numerical), 1e-8);
    assert(relErr < 1e-4, `x=${x} analytical=${analytical} numerical=${numerical} relErr=${relErr}`);
  }
});

// --- Forward Pass ---
console.log('\nForward Pass:');
test('produces valid probability distribution', () => {
  const model = createModel();
  const { probs } = forward(model, [0, 1, 2, 3]);
  assert(probs.length === NUM_TOKENS, `length=${probs.length}`);
  const sum = probs.reduce((a, b) => a + b, 0);
  assert(approxEqual(sum, 1.0, 1e-6), `sum=${sum}`);
  for (const p of probs) {
    assert(p >= 0, `negative prob: ${p}`);
    assert(!isNaN(p), 'NaN in probs');
  }
});

test('works with various sequence lengths', () => {
  const model = createModel();
  for (const len of [1, 2, 5, 12, 24]) {
    const tokens = Array.from({ length: len }, (_, i) => i % 16);
    const { probs } = forward(model, tokens);
    const sum = probs.reduce((a, b) => a + b, 0);
    assert(approxEqual(sum, 1.0, 1e-6), `len=${len} sum=${sum}`);
  }
});

// --- Data Generation ---
console.log('\nData Generation:');
test('generates valid waves', () => {
  const waves = generateWaves(30);
  assert(waves.length === 30, `count=${waves.length}`);
  for (const wave of waves) {
    assert(wave.length >= 60 && wave.length <= 100, `length=${wave.length}`);
    for (const token of wave) {
      assert(token >= 0 && token <= 15, `token out of range: ${token}`);
      assert(Number.isInteger(token), `non-integer token: ${token}`);
    }
  }
});

test('samples valid training examples', () => {
  const waves = generateWaves(30);
  for (let i = 0; i < 100; i++) {
    const { input, target } = sampleTrainingExample(waves);
    assert(input.length >= 3, `input too short: ${input.length}`);
    assert(input.length <= SEQ_LEN, `input too long: ${input.length}`);
    assert(target >= 0 && target <= 15, `target out of range: ${target}`);
  }
});

// --- Gradient Verification ---
console.log('\nGradient Verification (analytical vs numerical):');
test('gradients match for all parameters', () => {
  // Use a fixed seed by creating a small model and fixed input
  const model = createModel();
  const input = [5, 5, 5, 5];
  const target = 5;

  const { probs, cache } = forward(model, input);
  const grads = backward(model, cache, target);

  // Check matrix parameters
  const matParams = ['tok_emb', 'pos_emb', 'Wq', 'Wk', 'Wv', 'Wo', 'ffn_up', 'ffn_down', 'out_proj'];
  let maxRelErr = 0;
  let totalChecked = 0;
  let errorsAboveThreshold = 0;

  for (const name of matParams) {
    const mat = model[name];
    const grad = grads[name];
    // Check a random subset for speed
    const numChecks = Math.min(5, mat.length);
    for (let c = 0; c < numChecks; c++) {
      const i = Math.floor(Math.random() * mat.length);
      const j = Math.floor(Math.random() * mat[i].length);
      const ng = numericalGradient(model, input, target, name, i, j);
      const ag = grad[i][j];
      const absErr = Math.abs(ng - ag);
      const relErr = absErr / Math.max(Math.abs(ng), Math.abs(ag), 1e-8);
      totalChecked++;
      if (relErr > maxRelErr) maxRelErr = relErr;
      // Allow near-zero gradients (absErr < 1e-7) even if relative error is high
      if (relErr > 1e-3 && absErr > 1e-7) {
        errorsAboveThreshold++;
        console.log(`    WARNING: ${name}[${i}][${j}] numerical=${ng.toFixed(8)} analytical=${ag.toFixed(8)} relErr=${relErr.toFixed(6)} absErr=${absErr.toExponential(2)}`);
      }
    }
  }

  // Check bias parameters
  const biasParams = ['ffn_up_bias', 'ffn_down_bias'];
  for (const name of biasParams) {
    const grad = grads[name];
    const numChecks = Math.min(5, grad.length);
    for (let c = 0; c < numChecks; c++) {
      const i = Math.floor(Math.random() * grad.length);
      const ng = numericalGradient(model, input, target, name, i, 0);
      const ag = grad[i];
      const absErr = Math.abs(ng - ag);
      const relErr = absErr / Math.max(Math.abs(ng), Math.abs(ag), 1e-8);
      totalChecked++;
      if (relErr > maxRelErr) maxRelErr = relErr;
      if (relErr > 1e-3 && absErr > 1e-7) {
        errorsAboveThreshold++;
        console.log(`    WARNING: ${name}[${i}] numerical=${ng.toFixed(8)} analytical=${ag.toFixed(8)} relErr=${relErr.toFixed(6)}`);
      }
    }
  }

  console.log(`    Checked ${totalChecked} parameters, max relative error: ${maxRelErr.toFixed(8)}`);
  assert(errorsAboveThreshold === 0, `${errorsAboveThreshold} parameters exceeded error threshold of 1e-3`);
});

// Broader gradient check with random input
test('gradients match for random input', () => {
  const model = createModel();
  const input = [3, 7, 11, 2, 9];
  const target = 8;

  const { probs, cache } = forward(model, input);
  const grads = backward(model, cache, target);

  const matParams = ['tok_emb', 'Wq', 'Wk', 'Wv', 'Wo', 'ffn_up', 'ffn_down', 'out_proj'];
  let errorsAboveThreshold = 0;
  let maxRelErr = 0;

  for (const name of matParams) {
    const mat = model[name];
    const grad = grads[name];
    for (let c = 0; c < 3; c++) {
      const i = Math.floor(Math.random() * mat.length);
      const j = Math.floor(Math.random() * mat[i].length);
      const ng = numericalGradient(model, input, target, name, i, j);
      const ag = grad[i][j];
      const absErr = Math.abs(ng - ag);
      const relErr = absErr / Math.max(Math.abs(ng), Math.abs(ag), 1e-8);
      if (relErr > maxRelErr) maxRelErr = relErr;
      if (relErr > 1e-3 && absErr > 1e-7) {
        errorsAboveThreshold++;
        console.log(`    WARNING: ${name}[${i}][${j}] numerical=${ng.toFixed(8)} analytical=${ag.toFixed(8)} relErr=${relErr.toFixed(6)}`);
      }
    }
  }

  console.log(`    Max relative error: ${maxRelErr.toFixed(8)}`);
  assert(errorsAboveThreshold === 0, `${errorsAboveThreshold} parameters exceeded error threshold`);
});

// --- Summary ---
console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
process.exit(failed > 0 ? 1 : 0);
