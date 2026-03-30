// Drums Transformer Demo — React component
// 16 tokens = 4-bit bitmask: kick(1) snare(2) closed-hat(4) open-hat(8)

const { useState, useRef, useEffect, useCallback, useMemo } = React;

var DRUM_LABELS = ['K', 'S', 'H', 'O'];
var DRUM_COLORS = ['#e44', '#4af', '#4e4', '#fa4'];
var DRUM_BITS = [1, 2, 4, 8];

// Web Audio synthesis
function createDrumSynth() {
  var ctx = null;
  function getCtx() {
    if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
    return ctx;
  }
  return {
    play: function(token) {
      var ac = getCtx();
      var now = ac.currentTime;
      if (token & 1) { // kick
        var osc = ac.createOscillator(); var g = ac.createGain();
        osc.type = 'sine'; osc.frequency.setValueAtTime(150, now); osc.frequency.exponentialRampToValueAtTime(40, now + 0.12);
        g.gain.setValueAtTime(0.8, now); g.gain.exponentialRampToValueAtTime(0.001, now + 0.2);
        osc.connect(g); g.connect(ac.destination); osc.start(now); osc.stop(now + 0.2);
      }
      if (token & 2) { // snare
        var bufSize = ac.sampleRate * 0.1; var buf = ac.createBuffer(1, bufSize, ac.sampleRate);
        var d = buf.getChannelData(0); for (var i = 0; i < bufSize; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / bufSize, 3);
        var src = ac.createBufferSource(); var g = ac.createGain();
        src.buffer = buf; g.gain.setValueAtTime(0.6, now); g.gain.exponentialRampToValueAtTime(0.001, now + 0.12);
        src.connect(g); g.connect(ac.destination); src.start(now);
        // Snare body tone
        var osc = ac.createOscillator(); var g2 = ac.createGain();
        osc.type = 'triangle'; osc.frequency.setValueAtTime(200, now);
        g2.gain.setValueAtTime(0.3, now); g2.gain.exponentialRampToValueAtTime(0.001, now + 0.08);
        osc.connect(g2); g2.connect(ac.destination); osc.start(now); osc.stop(now + 0.08);
      }
      if (token & 4) { // closed hat
        var bufSize = ac.sampleRate * 0.04; var buf = ac.createBuffer(1, bufSize, ac.sampleRate);
        var d = buf.getChannelData(0); for (var i = 0; i < bufSize; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / bufSize, 6);
        var src = ac.createBufferSource(); var hp = ac.createBiquadFilter(); var g = ac.createGain();
        src.buffer = buf; hp.type = 'highpass'; hp.frequency.value = 7000;
        g.gain.setValueAtTime(0.3, now); g.gain.exponentialRampToValueAtTime(0.001, now + 0.05);
        src.connect(hp); hp.connect(g); g.connect(ac.destination); src.start(now);
      }
      if (token & 8) { // open hat
        var bufSize = ac.sampleRate * 0.15; var buf = ac.createBuffer(1, bufSize, ac.sampleRate);
        var d = buf.getChannelData(0); for (var i = 0; i < bufSize; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / bufSize, 2);
        var src = ac.createBufferSource(); var hp = ac.createBiquadFilter(); var g = ac.createGain();
        src.buffer = buf; hp.type = 'highpass'; hp.frequency.value = 6000;
        g.gain.setValueAtTime(0.35, now); g.gain.exponentialRampToValueAtTime(0.001, now + 0.18);
        src.connect(hp); hp.connect(g); g.connect(ac.destination); src.start(now);
      }
    }
  };
}

function DrumTransformer() {
  var modelRef = useRef(null);
  var adamRef = useRef(null);
  var patternsRef = useRef(null);
  var epochRef = useRef(0);
  var trainIntervalRef = useRef(null);
  var playIntervalRef = useRef(null);
  var synthRef = useRef(null);
  var trailBufsRef = useRef({});
  var textureCanvasRefs = useRef({});

  var [phase, setPhase] = useState('EDIT'); // EDIT | IDLE | PLAYING | PAUSED
  var [isTraining, setIsTraining] = useState(false);
  var [seed, setSeed] = useState([]);
  var [generated, setGenerated] = useState([]);
  var [loss, setLoss] = useState(null);
  var [epoch, setEpoch] = useState(0);
  var [trail, setTrail] = useState(false);
  var [renderTick, setRenderTick] = useState(0);
  var lossHistoryRef = useRef([]);
  var [lr, setLr] = useState(0.001);
  var [dim, setDim] = useState(24);
  var [temp, setTemp] = useState(0.0);
  var [paramCount, setParamCount] = useState(0);
  var [playHead, setPlayHead] = useState(-1);
  var [bpm, setBpm] = useState(120);

  useEffect(function() {
    var m = createModel({ dim: dim });
    modelRef.current = m;
    adamRef.current = createAdam(m);
    patternsRef.current = generateDrumPatterns(30);
    synthRef.current = createDrumSynth();
    setParamCount(countParams(m));
  }, []);

  // --- Seed editing: toggle drum bits in a 16-step grid ---
  var [editSteps, setEditSteps] = useState(function() { return new Array(16).fill(0); });

  var toggleBit = useCallback(function(step, bit) {
    if (phase !== 'EDIT') return;
    setEditSteps(function(prev) {
      var next = prev.slice();
      next[step] ^= bit;
      return next;
    });
  }, [phase]);

  var confirmSeed = useCallback(function() {
    var s = editSteps.filter(function(_, i) { return i < 16; });
    if (s.length < 3) return;
    setSeed(s);
    setGenerated([]);
    setPhase('IDLE');
  }, [editSteps]);

  // --- Training ---
  var lrRef = useRef(lr); lrRef.current = lr;

  var startTraining = useCallback(function() {
    if (trainIntervalRef.current) return;
    setIsTraining(true);
    trainIntervalRef.current = setInterval(function() {
      var model = modelRef.current, adam = adamRef.current, patterns = patternsRef.current;
      if (!model || !adam || !patterns) return;
      var totalLoss = 0, count = 0;
      for (var i = 0; i < 24; i++) {
        var s = sampleExample(patterns, model.seqLen);
        var r = trainStep(model, adam, s.input, s.target, lrRef.current);
        if (!isNaN(r.loss) && r.loss <= 20) { totalLoss += r.loss; count++; }
      }
      epochRef.current++;
      var avgLoss = count > 0 ? totalLoss / count : null;
      if (avgLoss !== null) lossHistoryRef.current.push(avgLoss);
      setLoss(avgLoss); setEpoch(epochRef.current); setRenderTick(function(n) { return n + 1; });
    }, 50);
  }, []);

  var stopTraining = useCallback(function() {
    if (trainIntervalRef.current) { clearInterval(trainIntervalRef.current); trainIntervalRef.current = null; }
    setIsTraining(false);
  }, []);

  // --- Generation ---
  var tempRef = useRef(temp); tempRef.current = temp;
  var seedRef = useRef(seed); seedRef.current = seed;

  var generateOne = useCallback(function() {
    var model = modelRef.current; if (!model) return null;
    var currentSeed = seedRef.current;
    var next = null;
    setGenerated(function(prev) {
      var full = currentSeed.concat(prev);
      var context = full.slice(-model.seqLen);
      var result = forward(model, context);
      next = sampleWithTemp(result.logits, tempRef.current);
      return prev.concat([next]);
    });
    setRenderTick(function(n) { return n + 1; });
    return next;
  }, []);

  // --- Playback with audio ---
  var bpmRef = useRef(bpm); bpmRef.current = bpm;

  var startPlaying = useCallback(function() {
    if (playIntervalRef.current) return;
    setPhase('PLAYING');
    var stepIndex = 0;
    var allSteps = seedRef.current.concat([]);
    // Pre-generate enough steps
    var model = modelRef.current;
    if (model) {
      var genSoFar = [];
      setGenerated(function(prev) { genSoFar = prev.slice(); return prev; });
      allSteps = seedRef.current.concat(genSoFar);
    }

    function tick() {
      setGenerated(function(prev) {
        var full = seedRef.current.concat(prev);
        // Generate more if we need them
        while (full.length <= stepIndex + seedRef.current.length) {
          var model = modelRef.current; if (!model) break;
          var context = full.slice(-model.seqLen);
          var result = forward(model, context);
          var next = sampleWithTemp(result.logits, tempRef.current);
          prev = prev.concat([next]);
          full = seedRef.current.concat(prev);
        }
        // Play current step
        var idx = stepIndex % full.length;
        var token = full[idx];
        if (synthRef.current && token > 0) synthRef.current.play(token);
        setPlayHead(idx);
        stepIndex++;
        return prev;
      });
    }

    tick(); // play first step immediately
    var ms = 60000 / bpmRef.current / 4; // 16th note interval
    playIntervalRef.current = setInterval(function() {
      var ms2 = 60000 / bpmRef.current / 4;
      tick();
      // If BPM changed, restart interval
      if (Math.abs(ms2 - ms) > 1) {
        clearInterval(playIntervalRef.current);
        ms = ms2;
        playIntervalRef.current = setInterval(tick, ms);
      }
    }, ms);
  }, []);

  var stopPlaying = useCallback(function() {
    if (playIntervalRef.current) { clearInterval(playIntervalRef.current); playIntervalRef.current = null; }
    setPlayHead(-1);
  }, []);

  var handlePlay = useCallback(function() {
    if (phase === 'PLAYING') { stopPlaying(); setPhase('PAUSED'); }
    else { startPlaying(); }
  }, [phase, startPlaying, stopPlaying]);

  var handleRewind = useCallback(function() {
    stopPlaying(); setGenerated([]); setPhase('IDLE'); setPlayHead(-1);
  }, [stopPlaying]);

  var handleClear = useCallback(function() {
    stopPlaying(); setSeed([]); setGenerated([]);
    setEditSteps(new Array(16).fill(0)); setPhase('EDIT'); setPlayHead(-1);
  }, [stopPlaying]);

  var handleReset = useCallback(function() {
    stopPlaying(); stopTraining();
    var m = createModel({ dim: dim }); modelRef.current = m; adamRef.current = createAdam(m);
    patternsRef.current = generateDrumPatterns(30); epochRef.current = 0;
    trailBufsRef.current = {}; lossHistoryRef.current = [];
    setSeed([]); setGenerated([]); setEditSteps(new Array(16).fill(0));
    setLoss(null); setEpoch(0); setParamCount(countParams(m)); setPhase('EDIT'); setPlayHead(-1);
  }, [stopPlaying, stopTraining, dim]);

  useEffect(function() { return function() { stopPlaying(); stopTraining(); }; }, [stopPlaying, stopTraining]);

  // --- Textures ---
  var textures = useMemo(function() {
    var m = modelRef.current; if (!m) return [];
    var d = m.dim, fd = m.ffnDim;
    return [
      { name: 'tok embed', data: m.tok_emb, rows: m.numTokens, cols: d },
      { name: 'pos embed', data: m.pos_emb, rows: m.seqLen, cols: d },
      { name: 'W_query', data: m.Wq, rows: d, cols: d }, { name: 'W_key', data: m.Wk, rows: d, cols: d },
      { name: 'W_value', data: m.Wv, rows: d, cols: d }, { name: 'W_out', data: m.Wo, rows: d, cols: d },
      { name: 'FFN up', data: m.ffn_up, rows: fd, cols: d }, { name: 'FFN down', data: m.ffn_down, rows: d, cols: fd },
      { name: 'readout', data: m.out_proj, rows: m.numTokens, cols: d },
    ];
  }, [epoch, generated.length, renderTick]);

  useEffect(function() {
    for (var tex of textures) {
      var canvas = textureCanvasRefs.current[tex.name]; if (!canvas) continue;
      if (!trailBufsRef.current[tex.name]) trailBufsRef.current[tex.name] = new Float64Array(tex.rows * tex.cols);
      renderTexture(canvas, tex.data, tex.rows, tex.cols, trail, trail ? trailBufsRef.current[tex.name] : null);
    }
  }, [textures, trail]);

  // --- Render ---
  var full = seed.concat(generated);
  var btnStyle = function(active, disabled) {
    return { padding: '6px 10px', border: 'none', borderRadius: 4, cursor: disabled ? 'default' : 'pointer', fontSize: 13, fontFamily: 'monospace', fontWeight: 600, background: active ? '#c22' : disabled ? '#333' : '#444', color: disabled ? '#666' : '#eee', opacity: disabled ? 0.5 : 1 };
  };

  // Loss canvas
  var lossCanvasRef = useRef(null);
  useEffect(function() {
    var canvas = lossCanvasRef.current; if (!canvas) return;
    var ctx2d = canvas.getContext('2d');
    var w = canvas.width, h = canvas.height;
    ctx2d.fillStyle = '#111'; ctx2d.fillRect(0, 0, w, h);
    var history = lossHistoryRef.current;
    if (history.length < 2) {
      ctx2d.fillStyle = 'rgba(255,255,255,0.3)'; ctx2d.textAlign = 'center'; ctx2d.font = '12px monospace';
      ctx2d.fillText(isTraining ? 'training...' : 'tap grid below to create a beat', w / 2, h / 2);
      return;
    }
    var maxL = Math.max.apply(null, history.slice(0, 20)), minL = Math.min.apply(null, history);
    var yR = Math.max(maxL - minL, 0.01);
    var pad = { top: 12, bottom: 16, left: 40, right: 8 };
    var gw = w - pad.left - pad.right, gh = h - pad.top - pad.bottom;
    ctx2d.strokeStyle = '#f84'; ctx2d.lineWidth = 1.5; ctx2d.beginPath();
    for (var i = 0; i < history.length; i++) {
      var x = pad.left + (i / (history.length - 1)) * gw;
      var y = pad.top + ((maxL - history[i]) / yR) * gh;
      if (i === 0) ctx2d.moveTo(x, y); else ctx2d.lineTo(x, y);
    }
    ctx2d.stroke();
    ctx2d.fillStyle = '#f84'; ctx2d.font = '10px monospace'; ctx2d.textAlign = 'left';
    ctx2d.fillText('loss ' + history[history.length - 1].toFixed(4), pad.left + 4, pad.top - 2);
    ctx2d.fillStyle = '#666'; ctx2d.textAlign = 'center'; ctx2d.fillText('ep ' + history.length, w / 2, h - 2);
  }, [epoch, isTraining]);

  // Resize loss canvas
  useEffect(function() {
    var canvas = lossCanvasRef.current; if (!canvas) return;
    var resize = function() { var p = canvas.parentElement; if (p) { canvas.width = p.clientWidth; canvas.height = p.clientHeight; } };
    resize(); var ro = new ResizeObserver(resize); ro.observe(canvas.parentElement);
    return function() { ro.disconnect(); };
  }, []);

  return (
    <div style={{ background: '#0a0a0a', color: '#eee', fontFamily: 'monospace', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{ padding: '8px 12px', borderBottom: '1px solid #333', fontSize: 13, color: '#888' }}>
        <span style={{ color: '#eee', fontWeight: 700 }}>drum.transformer</span>
        {' '}{paramCount.toLocaleString()} params · 16 combos · {dim}d
      </div>

      {/* Toolbar */}
      <div style={{ padding: '6px 12px', display: 'flex', gap: 6, flexWrap: 'wrap', borderBottom: '1px solid #333', alignItems: 'center' }}>
        <button style={btnStyle(isTraining)} onClick={isTraining ? stopTraining : startTraining}>{isTraining ? '■ Stop' : '⟳ Train'}</button>
        {phase === 'EDIT' ? (
          <button style={btnStyle(false, editSteps.every(function(s) { return s === 0; }))} onClick={confirmSeed} disabled={editSteps.every(function(s) { return s === 0; })}>✓ Set Seed</button>
        ) : (
          <button style={btnStyle(phase === 'PLAYING')} onClick={handlePlay}>{phase === 'PLAYING' ? '❚❚ Pause' : '▶ Play'}</button>
        )}
        <button style={btnStyle(false, phase === 'EDIT')} onClick={handleRewind} disabled={phase === 'EDIT'}>↺ Rewind</button>
        <button style={btnStyle()} onClick={handleClear}>✕ Clear</button>
        <div style={{ flex: 1 }} />
        <button style={{ ...btnStyle(), background: '#622' }} onClick={handleReset}>RESET</button>
      </div>

      {/* Status bar */}
      <div style={{ padding: '6px 12px', display: 'flex', gap: 10, alignItems: 'center', borderBottom: '1px solid #333', fontSize: 11, flexWrap: 'wrap' }}>
        <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', border: '2px solid #888', background: trail ? '#4f4' : 'transparent', display: 'inline-block' }} onClick={function() { setTrail(function(t) { return !t; }); }} />TRAIL
        </label>
        <span>dim</span><input type="range" min={8} max={48} step={4} value={dim} onChange={function(e) { setDim(+e.target.value); }} disabled={isTraining || phase !== 'EDIT'} style={{ width: 50 }} /><span>{dim}</span>
        <span style={{ color: '#555' }}>|</span>
        <span>lr</span><input type="range" min={-4} max={-1} step={0.1} value={Math.log10(lr)} onChange={function(e) { setLr(Math.round(Math.pow(10, +e.target.value) * 10000) / 10000); }} style={{ width: 50 }} /><span>{lr}</span>
        <span style={{ color: '#555' }}>|</span>
        <span>temp</span><input type="range" min={0} max={1.5} step={0.05} value={temp} onChange={function(e) { setTemp(+e.target.value); }} style={{ width: 50 }} /><span>{temp.toFixed(2)}</span>
        <span style={{ color: '#555' }}>|</span>
        <span>bpm</span><input type="range" min={60} max={200} step={5} value={bpm} onChange={function(e) { setBpm(+e.target.value); }} style={{ width: 50 }} /><span>{bpm}</span>
        <span style={{ color: '#555' }}>|</span>
        <span>ep {epoch}</span><span>loss {loss !== null ? loss.toFixed(4) : '—'}</span><span>{generated.length} gen</span>
      </div>

      {/* Loss graph */}
      <div style={{ height: 60, background: '#111', borderBottom: '1px solid #222' }}>
        <canvas ref={lossCanvasRef} style={{ width: '100%', height: '100%' }} />
      </div>

      {/* Beat Grid */}
      <div style={{ padding: '12px', overflowX: 'auto' }}>
        {phase === 'EDIT' ? (
          /* Editing grid: 4 rows (K,S,H,O) x 16 steps */
          <div>
            <div style={{ fontSize: 10, color: '#666', marginBottom: 6 }}>tap cells to toggle · then press "Set Seed"</div>
            <div style={{ display: 'flex', gap: 2 }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2, marginRight: 4 }}>
                {DRUM_LABELS.map(function(label, row) { return <div key={row} style={{ height: 28, display: 'flex', alignItems: 'center', fontSize: 11, color: DRUM_COLORS[row], width: 16 }}>{label}</div>; })}
              </div>
              {editSteps.map(function(token, col) {
                return <div key={col} style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {DRUM_BITS.map(function(bit, row) {
                    var on = (token & bit) !== 0;
                    return <div key={row} onClick={function() { toggleBit(col, bit); }}
                      style={{ width: 28, height: 28, borderRadius: 3, cursor: 'pointer', border: '1px solid #333',
                        background: on ? DRUM_COLORS[row] : (col % 4 === 0 ? '#1a1a1a' : '#111'),
                        opacity: on ? 1 : 0.6 }} />;
                  })}
                </div>;
              })}
            </div>
          </div>
        ) : (
          /* Playback grid: show seed (white borders) + generated (colored borders) */
          <div style={{ overflowX: 'auto' }}>
            <div style={{ display: 'flex', gap: 1, minWidth: 'min-content' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 1, marginRight: 4, flexShrink: 0 }}>
                {DRUM_LABELS.map(function(label, row) { return <div key={row} style={{ height: 22, display: 'flex', alignItems: 'center', fontSize: 10, color: DRUM_COLORS[row], width: 16 }}>{label}</div>; })}
              </div>
              {full.map(function(token, col) {
                var isSeed = col < seed.length;
                var isHead = col === playHead;
                return <div key={col} style={{ display: 'flex', flexDirection: 'column', gap: 1, flexShrink: 0 }}>
                  {DRUM_BITS.map(function(bit, row) {
                    var on = (token & bit) !== 0;
                    return <div key={row} style={{
                      width: 22, height: 22, borderRadius: 2,
                      border: isHead ? '2px solid #fff' : isSeed ? '1px solid #555' : '1px solid #333',
                      background: on ? DRUM_COLORS[row] : '#111',
                      opacity: on ? 1 : 0.3
                    }} />;
                  })}
                </div>;
              })}
            </div>
            {seed.length > 0 && generated.length > 0 && (
              <div style={{ fontSize: 10, color: '#666', marginTop: 4 }}>
                <span style={{ color: '#888' }}>seed ({seed.length})</span> | <span style={{ color: '#4f4' }}>generated ({generated.length})</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Texture Panel */}
      <div style={{ flex: 1, overflow: 'auto', padding: 8, borderTop: '1px solid #222' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 8 }}>
          {textures.map(function(tex) {
            return <div key={tex.name} style={{ textAlign: 'center' }}>
              <canvas ref={function(el) { if (el) textureCanvasRefs.current[tex.name] = el; }} style={{ imageRendering: 'pixelated', maxWidth: '100%' }} />
              <div style={{ fontSize: 10, color: '#888', marginTop: 2 }}>{tex.name} {tex.rows}×{tex.cols}</div>
            </div>;
          })}
        </div>
      </div>
    </div>
  );
}
