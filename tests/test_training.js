const {
  createModel, forward, crossEntropyLoss, trainStep,
  generateWaves, sampleTrainingExample,
} = require('../src/model.js');

console.log('=== Training Tests ===\n');

// Test 1: Trivial task — learn to predict a constant
console.log('Test 1: Learn to predict constant [5,5,5,5] → 5');
{
  const model = createModel();
  const input = [5, 5, 5, 5];
  const target = 5;
  let loss;
  for (let i = 0; i < 200; i++) {
    loss = trainStep(model, input, target, 0.01);
    if (i % 50 === 0) console.log(`  epoch ${i}: loss=${loss.toFixed(4)}`);
  }
  console.log(`  final loss: ${loss.toFixed(4)}`);
  const { probs } = forward(model, input);
  const predicted = probs.indexOf(Math.max(...probs));
  console.log(`  predicted: ${predicted}, target: ${target}, prob: ${probs[target].toFixed(4)}`);
  console.log(`  ${loss < 0.5 ? '✓' : '✗'} loss < 0.5: ${loss.toFixed(4)}`);
  console.log(`  ${predicted === target ? '✓' : '✗'} prediction correct`);
}

// Test 2: Training on sine waves — loss should decrease from ~2.77
console.log('\nTest 2: Training on sine wave data (1000 ticks, batch=24 per tick)');
{
  const model = createModel();
  let waves = generateWaves(30);
  const losses = [];

  for (let tick = 0; tick < 1000; tick++) {
    let tickLoss = 0;
    for (let b = 0; b < 24; b++) {
      const { input, target } = sampleTrainingExample(waves);
      const loss = trainStep(model, input, target, 0.01);
      if (!isNaN(loss)) tickLoss += loss;
    }
    tickLoss /= 24;
    losses.push(tickLoss);
    if (tick % 200 === 0) console.log(`  tick ${tick}: loss=${tickLoss.toFixed(4)}`);
    if (tick === 300) {
      waves = generateWaves(30);
      console.log('  (regenerated wave corpus at tick 300)');
    }
    if (tick === 600) {
      waves = generateWaves(30);
      console.log('  (regenerated wave corpus at tick 600)');
    }
    if (tick === 900) {
      waves = generateWaves(30);
    }
  }

  const firstAvg = losses.slice(0, 20).reduce((a, b) => a + b) / 20;
  const lastAvg = losses.slice(-20).reduce((a, b) => a + b) / 20;
  console.log(`  first 20 avg: ${firstAvg.toFixed(4)}, last 20 avg: ${lastAvg.toFixed(4)}`);
  console.log(`  ${lastAvg < firstAvg ? '✓' : '✗'} loss decreased`);
  console.log(`  ${lastAvg < 1.5 ? '✓' : '~'} last avg < 1.5: ${lastAvg.toFixed(4)}`);

  // Test generation after training
  console.log('\n  Generation test:');
  const seed = [7, 8, 9, 10, 11];
  const generated = [...seed];
  for (let i = 0; i < 15; i++) {
    const { probs } = forward(model, generated.slice(-24));
    const next = probs.indexOf(Math.max(...probs));
    generated.push(next);
  }
  console.log(`  seed: [${seed.join(',')}]`);
  console.log(`  generated: [${generated.join(',')}]`);
}

console.log('\n=== Training Tests Complete ===');
