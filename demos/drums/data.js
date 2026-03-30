// Drums Demo — Training data generation
// Vocabulary: 16 tokens = 4-bit bitmask of drum hits
//   bit 0 = kick, bit 1 = snare, bit 2 = closed hat, bit 3 = open hat
// Patterns are 16 or 32 steps (standard bar/2-bar loops)

var DRUM_NAMES = ['kick', 'snare', 'hat', 'open'];

// Common drum patterns (hand-crafted + procedural)
function generateDrumPatterns(count) {
  count = count || 30;
  var patterns = [];

  // --- Hand-crafted classic patterns ---
  // Basic 4-on-the-floor (house/disco)
  // K on 1,5,9,13  S on 5,13  H on every step
  patterns.push(makePattern([
    5,4,4,4, 7,4,4,4, 5,4,4,4, 7,4,4,4
  ]));

  // Standard rock beat
  // K on 1,9  S on 5,13  H on odd steps
  patterns.push(makePattern([
    5,4,0,4, 6,4,0,4, 5,4,0,4, 6,4,0,4
  ]));

  // Funk pattern
  patterns.push(makePattern([
    5,0,4,1, 6,0,4,0, 1,4,5,0, 6,4,0,4
  ]));

  // Reggaeton / dembow
  patterns.push(makePattern([
    5,0,4,2, 5,0,4,2, 5,0,4,2, 5,0,4,2
  ]));

  // Bossa nova feel
  patterns.push(makePattern([
    5,4,0,4, 4,6,4,0, 5,4,0,4, 0,6,4,4
  ]));

  // Breakbeat
  patterns.push(makePattern([
    5,0,4,0, 6,0,0,5, 0,4,6,0, 4,1,4,0
  ]));

  // Hi-hat focused (trap)
  patterns.push(makePattern([
    5,4,4,4, 6,4,4,4, 5,4,4,4, 6,4,4,4
  ]));
  patterns.push(makePattern([
    1,4,4,4, 2,4,4,4, 1,4,4,8, 2,4,4,4
  ]));

  // Fill the rest with procedural patterns
  while (patterns.length < count) {
    patterns.push(generateProceduralPattern());
  }

  return patterns;
}

function makePattern(steps) {
  // Repeat to make a longer sequence for window sampling
  var rep = [];
  for (var r = 0; r < 4; r++) for (var i = 0; i < steps.length; i++) rep.push(steps[i]);
  return rep;
}

function generateProceduralPattern() {
  var len = 16;
  var steps = new Array(len).fill(0);

  // Kick: mostly on beat (positions 0, 4, 8, 12) with some variation
  var kickStyle = Math.floor(Math.random() * 3);
  if (kickStyle === 0) { // 4-on-the-floor
    for (var i = 0; i < len; i += 4) steps[i] |= 1;
  } else if (kickStyle === 1) { // 1 and 3
    steps[0] |= 1; steps[8] |= 1;
    if (Math.random() < 0.5) steps[6] |= 1; // ghost kick
  } else { // syncopated
    steps[0] |= 1; steps[3] |= 1; steps[8] |= 1; steps[11] |= 1;
  }

  // Snare: typically on 2 and 4 (positions 4, 12)
  var snareStyle = Math.floor(Math.random() * 3);
  if (snareStyle === 0) { // standard backbeat
    steps[4] |= 2; steps[12] |= 2;
  } else if (snareStyle === 1) { // with ghost notes
    steps[4] |= 2; steps[12] |= 2;
    if (Math.random() < 0.6) steps[10] |= 2;
  } else { // offbeat snare
    steps[2] |= 2; steps[6] |= 2; steps[10] |= 2; steps[14] |= 2;
  }

  // Hi-hat: dense or sparse
  var hatStyle = Math.floor(Math.random() * 3);
  if (hatStyle === 0) { // 8th notes
    for (var i = 0; i < len; i += 2) steps[i] |= 4;
  } else if (hatStyle === 1) { // 16th notes
    for (var i = 0; i < len; i++) steps[i] |= 4;
  } else { // sparse/offbeat
    for (var i = 1; i < len; i += 2) steps[i] |= 4;
  }

  // Open hat: occasional (replace some closed hats)
  if (Math.random() < 0.5) {
    var openPos = [Math.floor(Math.random() * 4) * 4 + 2]; // off-beat position
    if (Math.random() < 0.3) openPos.push(Math.floor(Math.random() * 4) * 4 + 3);
    for (var p of openPos) {
      if (p < len) { steps[p] = (steps[p] & ~4) | 8; } // replace closed with open
    }
  }

  return makePattern(steps);
}

function sampleExample(patterns, seqLen) {
  seqLen = seqLen || 24;
  var p = patterns[Math.floor(Math.random() * patterns.length)];
  var st = Math.floor(Math.random() * (p.length - 3));
  var wl = 3 + Math.floor(Math.random() * (Math.min(seqLen, p.length - st - 1) - 2));
  return { input: p.slice(st, st + wl), target: p[st + wl] };
}

if (typeof module !== 'undefined') module.exports = { DRUM_NAMES: DRUM_NAMES, generateDrumPatterns: generateDrumPatterns, sampleExample: sampleExample };
