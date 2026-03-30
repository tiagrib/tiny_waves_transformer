// Wave Transformer Demo — React component
// Depends on shared/ (core, model, optimizer, ui) and demos/waves/data.js being in scope

const { useState, useRef, useEffect, useCallback, useMemo } = React;

function WaveTransformer() {
  const modelRef = useRef(null);
  const adamRef = useRef(null);
  const wavesRef = useRef(null);
  const epochRef = useRef(0);
  const trainIntervalRef = useRef(null);
  const genIntervalRef = useRef(null);
  const trailBufsRef = useRef({});
  const drawCanvasRef = useRef(null);
  const textureCanvasRefs = useRef({});
  const lastAttnRef = useRef(null);
  const lastFfnActRef = useRef(null);
  const lastProbsRef = useRef(null);

  const [phase, setPhase] = useState('DRAW');
  const [isTraining, setIsTraining] = useState(false);
  const [seed, setSeed] = useState([]);
  const [generated, setGenerated] = useState([]);
  const [seedSteps, setSeedSteps] = useState(24);
  const [loss, setLoss] = useState(null);
  const [epoch, setEpoch] = useState(0);
  const [trail, setTrail] = useState(false);
  const [drawPoints, setDrawPoints] = useState([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [renderTick, setRenderTick] = useState(0);
  const lossHistoryRef = useRef([]);
  const [lr, setLr] = useState(0.001);
  const [dim, setDim] = useState(24);
  const [temp, setTemp] = useState(0.0);
  const [paramCount, setParamCount] = useState(0);

  useEffect(() => {
    const m = createModel({ dim: dim });
    modelRef.current = m;
    adamRef.current = createAdam(m);
    wavesRef.current = generateWaves(30);
    setParamCount(countParams(m));
  }, []);

  const canvasToNorm = useCallback((canvas, e) => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
  }, []);

  const quantizeDraw = useCallback((points, steps) => {
    if (points.length < 2) return [];
    const sorted = [...points].sort((a, b) => a.x - b.x);
    const xMin = sorted[0].x, xMax = sorted[sorted.length - 1].x;
    if (xMax - xMin < 0.01) return [];
    const tokens = [];
    for (let s = 0; s < steps; s++) {
      const tx = xMin + (s / (steps - 1)) * (xMax - xMin);
      let best = sorted[0], bestDist = Math.abs(sorted[0].x - tx);
      for (const p of sorted) { const d = Math.abs(p.x - tx); if (d < bestDist) { best = p; bestDist = d; } }
      tokens.push(Math.max(0, Math.min(15, Math.round((1 - best.y) * 15))));
    }
    return tokens;
  }, []);

  const handlePointerDown = useCallback((e) => {
    if (phase !== 'DRAW') return;
    const canvas = drawCanvasRef.current;
    if (!canvas) return;
    canvas.setPointerCapture(e.pointerId);
    setIsDrawing(true);
    setDrawPoints([canvasToNorm(canvas, e)]);
  }, [phase, canvasToNorm]);

  const handlePointerMove = useCallback((e) => {
    if (!isDrawing || phase !== 'DRAW') return;
    const canvas = drawCanvasRef.current;
    if (!canvas) return;
    setDrawPoints(prev => [...prev, canvasToNorm(canvas, e)]);
  }, [isDrawing, phase, canvasToNorm]);

  const handlePointerUp = useCallback(() => {
    if (!isDrawing) return;
    setIsDrawing(false);
    setDrawPoints(prev => {
      const tokens = quantizeDraw(prev, seedSteps);
      if (tokens.length >= 3) { setSeed(tokens); setGenerated([]); setPhase('IDLE'); }
      return prev;
    });
  }, [isDrawing, quantizeDraw, seedSteps]);

  const drawWaveCanvas = useCallback(() => {
    const canvas = drawCanvasRef.current;
    if (!canvas) return;
    const ctx2d = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx2d.fillStyle = '#111';
    ctx2d.fillRect(0, 0, w, h);
    for (let level = 0; level <= 15; level++) {
      const y = h - (level / 15) * h;
      ctx2d.strokeStyle = level % 4 === 0 ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.07)';
      ctx2d.lineWidth = 1;
      ctx2d.beginPath(); ctx2d.moveTo(0, y); ctx2d.lineTo(w, y); ctx2d.stroke();
    }

    if (phase === 'DRAW') {
      const history = lossHistoryRef.current;
      if (drawPoints.length > 1) {
        ctx2d.strokeStyle = 'rgba(255,255,255,0.3)'; ctx2d.lineWidth = 2; ctx2d.beginPath();
        ctx2d.moveTo(drawPoints[0].x * w, drawPoints[0].y * h);
        for (let i = 1; i < drawPoints.length; i++) ctx2d.lineTo(drawPoints[i].x * w, drawPoints[i].y * h);
        ctx2d.stroke();
        const preview = quantizeDraw(drawPoints, seedSteps);
        if (preview.length > 0) {
          ctx2d.strokeStyle = 'white'; ctx2d.lineWidth = 2; ctx2d.beginPath();
          for (let i = 0; i < preview.length; i++) { const x = (i / (preview.length - 1)) * w, y = h - (preview[i] / 15) * h; if (i === 0) ctx2d.moveTo(x, y); else ctx2d.lineTo(x, y); }
          ctx2d.stroke();
          ctx2d.fillStyle = 'white';
          for (let i = 0; i < preview.length; i++) { const x = (i / (preview.length - 1)) * w, y = h - (preview[i] / 15) * h; ctx2d.beginPath(); ctx2d.arc(x, y, 3, 0, 2 * Math.PI); ctx2d.fill(); }
        }
      } else if (history.length > 1) {
        const pad = { top: 16, bottom: 20, left: 44, right: 12 };
        const gw = w - pad.left - pad.right, gh = h - pad.top - pad.bottom;
        const maxLoss = Math.max(...history.slice(0, 20)), minLoss = Math.min(...history);
        const yRange = Math.max(maxLoss - minLoss, 0.01);
        ctx2d.fillStyle = '#666'; ctx2d.font = '10px monospace'; ctx2d.textAlign = 'right';
        for (let i = 0; i <= 4; i++) { const v = minLoss + (yRange * (4 - i)) / 4, y = pad.top + (i / 4) * gh; ctx2d.fillText(v.toFixed(2), pad.left - 4, y + 3); ctx2d.strokeStyle = 'rgba(255,255,255,0.06)'; ctx2d.lineWidth = 1; ctx2d.beginPath(); ctx2d.moveTo(pad.left, y); ctx2d.lineTo(w - pad.right, y); ctx2d.stroke(); }
        ctx2d.fillStyle = '#666'; ctx2d.textAlign = 'center'; ctx2d.fillText('epoch ' + history.length, w / 2, h - 2);
        ctx2d.strokeStyle = '#f84'; ctx2d.lineWidth = 1.5; ctx2d.beginPath();
        for (let i = 0; i < history.length; i++) { const x = pad.left + (i / (history.length - 1)) * gw, y = pad.top + ((maxLoss - history[i]) / yRange) * gh; if (i === 0) ctx2d.moveTo(x, y); else ctx2d.lineTo(x, y); }
        ctx2d.stroke();
        ctx2d.fillStyle = '#f84'; ctx2d.textAlign = 'left'; ctx2d.font = '11px monospace';
        ctx2d.fillText('loss ' + history[history.length - 1].toFixed(4), pad.left + 4, pad.top - 4);
        if (!isTraining) { ctx2d.fillStyle = 'rgba(255,255,255,0.4)'; ctx2d.textAlign = 'center'; ctx2d.font = '13px monospace'; ctx2d.fillText('draw a seed wave', w / 2, h / 2); }
      } else {
        ctx2d.fillStyle = 'rgba(255,255,255,0.3)'; ctx2d.textAlign = 'center'; ctx2d.font = '13px monospace';
        ctx2d.fillText(isTraining ? 'training...' : 'draw a seed wave', w / 2, h / 2);
      }
    } else {
      const full = [...seed, ...generated]; if (full.length === 0) return;
      const xScale = w / Math.max(full.length - 1, 1);
      ctx2d.strokeStyle = 'white'; ctx2d.lineWidth = 2; ctx2d.beginPath();
      for (let i = 0; i < seed.length; i++) { const x = i * xScale, y = h - (seed[i] / 15) * h; if (i === 0) ctx2d.moveTo(x, y); else ctx2d.lineTo(x, y); }
      ctx2d.stroke(); ctx2d.fillStyle = 'white';
      for (let i = 0; i < seed.length; i++) { const x = i * xScale, y = h - (seed[i] / 15) * h; ctx2d.beginPath(); ctx2d.arc(x, y, 3, 0, 2 * Math.PI); ctx2d.fill(); }
      if (generated.length > 0) {
        const bx = (seed.length - 0.5) * xScale; ctx2d.setLineDash([4, 4]); ctx2d.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx2d.beginPath(); ctx2d.moveTo(bx, 0); ctx2d.lineTo(bx, h); ctx2d.stroke(); ctx2d.setLineDash([]);
        ctx2d.strokeStyle = '#4f4'; ctx2d.lineWidth = 2; ctx2d.beginPath();
        ctx2d.moveTo((seed.length - 1) * xScale, h - (seed[seed.length - 1] / 15) * h);
        for (let i = 0; i < generated.length; i++) { const x = (seed.length + i) * xScale, y = h - (generated[i] / 15) * h; ctx2d.lineTo(x, y); }
        ctx2d.stroke(); ctx2d.fillStyle = '#4f4';
        for (let i = 0; i < generated.length; i++) { const x = (seed.length + i) * xScale, y = h - (generated[i] / 15) * h; ctx2d.beginPath(); ctx2d.arc(x, y, 3, 0, 2 * Math.PI); ctx2d.fill(); }
      }
    }
  }, [phase, seed, generated, drawPoints, seedSteps, quantizeDraw, isTraining, epoch]);

  useEffect(() => {
    const canvas = drawCanvasRef.current; if (!canvas) return;
    const resize = () => { const parent = canvas.parentElement; if (parent) { canvas.width = parent.clientWidth; canvas.height = parent.clientHeight; } drawWaveCanvas(); };
    resize(); const ro = new ResizeObserver(resize); ro.observe(canvas.parentElement);
    return () => ro.disconnect();
  }, [drawWaveCanvas]);

  const startTraining = useCallback(() => {
    if (trainIntervalRef.current) return;
    setIsTraining(true);
    trainIntervalRef.current = setInterval(() => {
      const model = modelRef.current, adam = adamRef.current, waves = wavesRef.current;
      if (!model || !adam || !waves) return;
      let totalLoss = 0, count = 0;
      for (let i = 0; i < 24; i++) { const { input, target } = sampleExample(waves, model.seqLen); const { loss: l } = trainStep(model, adam, input, target, lrRef.current); if (!isNaN(l) && l <= 20) { totalLoss += l; count++; } }
      epochRef.current++;
      const avgLoss = count > 0 ? totalLoss / count : null;
      if (avgLoss !== null) lossHistoryRef.current.push(avgLoss);
      setLoss(avgLoss); setEpoch(epochRef.current); setRenderTick(n => n + 1);
    }, 50);
  }, []);

  const stopTraining = useCallback(() => { if (trainIntervalRef.current) { clearInterval(trainIntervalRef.current); trainIntervalRef.current = null; } setIsTraining(false); }, []);

  const lrRef = useRef(lr); lrRef.current = lr;
  const tempRef = useRef(temp); tempRef.current = temp;
  const seedRef = useRef(seed); seedRef.current = seed;

  const generateOne = useCallback(() => {
    const model = modelRef.current; if (!model) return;
    const currentSeed = seedRef.current;
    setGenerated(prev => {
      const full = [...currentSeed, ...prev];
      const context = full.slice(-model.seqLen);
      const { probs, logits, cache } = forward(model, context);
      const next = sampleWithTemp(logits, tempRef.current);
      lastAttnRef.current = cache.attnW; lastFfnActRef.current = cache.ffnAct; lastProbsRef.current = probs;
      return [...prev, next];
    });
    setRenderTick(n => n + 1);
  }, []);

  const startPlaying = useCallback(() => { if (genIntervalRef.current) return; setPhase('PLAYING'); genIntervalRef.current = setInterval(() => generateOne(), 150); }, [generateOne]);
  const stopPlaying = useCallback(() => { if (genIntervalRef.current) { clearInterval(genIntervalRef.current); genIntervalRef.current = null; } }, []);
  const handlePlay = useCallback(() => { if (phase === 'PLAYING') { stopPlaying(); setPhase('PAUSED'); } else { startPlaying(); } }, [phase, startPlaying, stopPlaying]);
  const handleStepBack = useCallback(() => { if (phase !== 'PAUSED' || generated.length === 0) return; setGenerated(prev => prev.slice(0, -1)); }, [phase, generated.length]);
  const handleStepForward = useCallback(() => { if (phase !== 'PAUSED') return; generateOne(); }, [phase, generateOne]);
  const handleRewind = useCallback(() => { stopPlaying(); setGenerated([]); lastAttnRef.current = null; lastFfnActRef.current = null; lastProbsRef.current = null; setPhase('IDLE'); }, [stopPlaying]);
  const handleClear = useCallback(() => { stopPlaying(); setSeed([]); setGenerated([]); setDrawPoints([]); lastAttnRef.current = null; lastFfnActRef.current = null; lastProbsRef.current = null; setPhase('DRAW'); }, [stopPlaying]);

  const handleReset = useCallback(() => {
    stopPlaying(); stopTraining();
    const m = createModel({ dim: dim }); modelRef.current = m; adamRef.current = createAdam(m);
    wavesRef.current = generateWaves(30); epochRef.current = 0; trailBufsRef.current = {}; lossHistoryRef.current = [];
    setSeed([]); setGenerated([]); setDrawPoints([]); setLoss(null); setEpoch(0); setParamCount(countParams(m));
    lastAttnRef.current = null; lastFfnActRef.current = null; lastProbsRef.current = null; setPhase('DRAW');
  }, [stopPlaying, stopTraining, dim]);

  useEffect(() => () => { stopPlaying(); stopTraining(); }, [stopPlaying, stopTraining]);

  const textures = useMemo(() => {
    const m = modelRef.current; if (!m) return [];
    const d = m.dim, fd = m.ffnDim;
    const list = [
      { name: 'tok embed', data: m.tok_emb, rows: m.numTokens, cols: d },
      { name: 'pos embed', data: m.pos_emb, rows: m.seqLen, cols: d },
      { name: 'W_query', data: m.Wq, rows: d, cols: d }, { name: 'W_key', data: m.Wk, rows: d, cols: d },
      { name: 'W_value', data: m.Wv, rows: d, cols: d }, { name: 'W_out', data: m.Wo, rows: d, cols: d },
      { name: 'FFN up', data: m.ffn_up, rows: fd, cols: d }, { name: 'FFN down', data: m.ffn_down, rows: d, cols: fd },
      { name: 'readout', data: m.out_proj, rows: m.numTokens, cols: d },
    ];
    if (lastAttnRef.current && generated.length > 0) {
      const aw = lastAttnRef.current, T = aw.length, padded = [];
      for (let i = 0; i < T; i++) { const row = new Float64Array(T); for (let j = 0; j <= i; j++) row[j] = aw[i][j]; padded.push(row); }
      list.push({ name: 'attention', data: padded, rows: T, cols: T });
    }
    if (lastFfnActRef.current && generated.length > 0) list.push({ name: 'FFN act', data: lastFfnActRef.current, rows: lastFfnActRef.current.length, cols: modelRef.current.ffnDim });
    return list;
  }, [epoch, generated.length, renderTick]);

  useEffect(() => { for (const tex of textures) { const canvas = textureCanvasRefs.current[tex.name]; if (!canvas) continue; if (!trailBufsRef.current[tex.name]) trailBufsRef.current[tex.name] = new Float64Array(tex.rows * tex.cols); renderTexture(canvas, tex.data, tex.rows, tex.cols, trail, trail ? trailBufsRef.current[tex.name] : null); } }, [textures, trail]);

  const probBar = lastProbsRef.current && generated.length > 0 ? lastProbsRef.current : null;
  const btnStyle = (active, disabled) => ({ padding: '6px 10px', border: 'none', borderRadius: 4, cursor: disabled ? 'default' : 'pointer', fontSize: 13, fontFamily: 'monospace', fontWeight: 600, background: active ? '#c22' : disabled ? '#333' : '#444', color: disabled ? '#666' : '#eee', opacity: disabled ? 0.5 : 1 });

  return (
    <div style={{ background: '#0a0a0a', color: '#eee', fontFamily: 'monospace', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '8px 12px', borderBottom: '1px solid #333', fontSize: 13, color: '#888' }}>
        <span style={{ color: '#eee', fontWeight: 700 }}>wave.transformer</span>
        {' '}{paramCount.toLocaleString()} params · 16 levels · {dim}d
      </div>
      <div style={{ padding: '6px 12px', display: 'flex', gap: 6, flexWrap: 'wrap', borderBottom: '1px solid #333', alignItems: 'center' }}>
        <button style={btnStyle(isTraining)} onClick={isTraining ? stopTraining : startTraining}>{isTraining ? '■ Stop' : '⟳ Train'}</button>
        <button style={btnStyle(phase === 'PLAYING', phase === 'DRAW')} onClick={handlePlay} disabled={phase === 'DRAW'}>{phase === 'PLAYING' ? '❚❚ Pause' : '▶ Play'}</button>
        <button style={btnStyle(false, phase !== 'PAUSED' || generated.length === 0)} onClick={handleStepBack} disabled={phase !== 'PAUSED' || generated.length === 0}>◀</button>
        <button style={btnStyle(false, phase !== 'PAUSED')} onClick={handleStepForward} disabled={phase !== 'PAUSED'}>▶</button>
        <button style={btnStyle(false, phase === 'DRAW')} onClick={handleRewind} disabled={phase === 'DRAW'}>↺ Rewind</button>
        <button style={btnStyle()} onClick={handleClear}>✕ Clear</button>
        <div style={{ flex: 1 }} />
        <button style={{ ...btnStyle(), background: '#622' }} onClick={handleReset}>RESET</button>
      </div>
      <div style={{ padding: '6px 12px', display: 'flex', gap: 10, alignItems: 'center', borderBottom: '1px solid #333', fontSize: 11, flexWrap: 'wrap' }}>
        <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}><span style={{ width: 10, height: 10, borderRadius: '50%', border: '2px solid #888', background: trail ? '#4f4' : 'transparent', display: 'inline-block' }} onClick={() => setTrail(t => !t)} />TRAIL</label>
        <span>seed</span><input type="range" min={8} max={48} value={seedSteps} onChange={e => setSeedSteps(+e.target.value)} disabled={phase !== 'DRAW'} style={{ width: 60 }} /><span>{seedSteps}</span>
        <span style={{ color: '#555' }}>|</span>
        <span>dim</span><input type="range" min={8} max={48} step={4} value={dim} onChange={e => setDim(+e.target.value)} disabled={isTraining || phase !== 'DRAW'} style={{ width: 50 }} /><span>{dim}</span>
        <span style={{ color: '#555' }}>|</span>
        <span>lr</span><input type="range" min={-4} max={-1} step={0.1} value={Math.log10(lr)} onChange={e => setLr(Math.round(10 ** +e.target.value * 10000) / 10000)} style={{ width: 50 }} /><span>{lr}</span>
        <span style={{ color: '#555' }}>|</span>
        <span>temp</span><input type="range" min={0} max={1.5} step={0.05} value={temp} onChange={e => setTemp(+e.target.value)} style={{ width: 50 }} /><span>{temp.toFixed(2)}</span>
        <span style={{ color: '#555' }}>|</span>
        <span>ep {epoch}</span><span>loss {loss !== null ? loss.toFixed(4) : '—'}</span><span>{generated.length} gen</span>
      </div>
      <div style={{ height: '25vh', minHeight: 120, background: '#111', position: 'relative' }}>
        <canvas ref={drawCanvasRef} style={{ width: '100%', height: '100%', touchAction: 'none', cursor: phase === 'DRAW' ? 'crosshair' : 'default' }} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={handlePointerUp} onPointerLeave={handlePointerUp} />
      </div>
      <div style={{ flex: 1, overflow: 'auto', padding: 8 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 8 }}>
          {textures.map(tex => (<div key={tex.name} style={{ textAlign: 'center' }}><canvas ref={el => { if (el) textureCanvasRefs.current[tex.name] = el; }} style={{ imageRendering: 'pixelated', maxWidth: '100%' }} /><div style={{ fontSize: 10, color: '#888', marginTop: 2 }}>{tex.name} {tex.rows}×{tex.cols}</div></div>))}
        </div>
        {probBar && (<div style={{ marginTop: 12 }}><div style={{ fontSize: 10, color: '#888', marginBottom: 4 }}>output probabilities</div><div style={{ display: 'flex', gap: 2, alignItems: 'end', height: 60 }}>{Array.from(probBar).map((p, i) => (<div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}><div style={{ width: '100%', background: '#4f4', height: Math.max(1, p * 60), borderRadius: 2 }} /><div style={{ fontSize: 8, color: '#666', marginTop: 2 }}>{i}</div></div>))}</div></div>)}
      </div>
    </div>
  );
}
