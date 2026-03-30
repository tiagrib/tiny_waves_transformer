// Build script: generates standalone HTML from shared core + demo files
// Usage: node build.js [demo-name|all]  (default: all)
const fs = require('fs');
const path = require('path');

const arg = process.argv[2] || 'all';

// Read shared modules (order matters)
const sharedFiles = ['core.js', 'model.js', 'optimizer.js', 'ui.js'];
function getSharedCode() {
  return sharedFiles.map(f => {
    let code = fs.readFileSync(path.join('shared', f), 'utf8');
    code = code.replace(/if \(typeof require[\s\S]*?\n\}\n?/g, '');
    code = code.replace(/if \(typeof module[^\n]*\n?/g, '');
    return '// --- shared/' + f + ' ---\n' + code.trim();
  }).join('\n\n');
}

function buildDemo(demo) {
  const demoDir = path.join('demos', demo);
  if (!fs.existsSync(demoDir)) { console.error('Demo not found: ' + demoDir); return false; }

  const sharedCode = getSharedCode();

  // Read demo-specific data.js
  let dataCode = '';
  const dataPath = path.join(demoDir, 'data.js');
  if (fs.existsSync(dataPath)) {
    dataCode = fs.readFileSync(dataPath, 'utf8');
    dataCode = dataCode.replace(/if \(typeof module[^\n]*\n?/g, '');
    dataCode = '// --- ' + demo + '/data.js ---\n' + dataCode.trim();
  }

  // Read demo JSX
  const jsxFiles = fs.readdirSync(demoDir).filter(f => f.endsWith('.jsx'));
  if (jsxFiles.length === 0) { console.error('No .jsx file in ' + demoDir); return false; }
  let jsxCode = fs.readFileSync(path.join(demoDir, jsxFiles[0]), 'utf8');
  jsxCode = jsxCode.replace(/^import .*$/gm, '');
  jsxCode = jsxCode.replace(/^export default /gm, '');
  // Find the main React component: last function with a PascalCase name
  const allFuncs = [...jsxCode.matchAll(/^function\s+([A-Z]\w+)/gm)];
  const componentName = allFuncs.length > 0 ? allFuncs[allFuncs.length - 1][1] : 'App';
  const title = demo.charAt(0).toUpperCase() + demo.slice(1) + ' Transformer';

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title}</title>
  <style>* { margin: 0; padding: 0; box-sizing: border-box; } body { background: #0a0a0a; }</style>
</head>
<body>
  <div id="root"></div>
  <script src="https://unpkg.com/react@18/umd/react.development.js"><\/script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"><\/script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"><\/script>
  <script type="text/babel">
${sharedCode}

${dataCode}

${jsxCode.trim()}

    ReactDOM.createRoot(document.getElementById('root')).render(
      React.createElement(${componentName})
    );
  <\/script>
</body>
</html>`;

  const outDir = path.join('dist', demo);
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, 'index.html'), html);
  console.log('  Built dist/' + demo + '/index.html');
  return true;
}

// Discover all demos (directories under demos/ that contain a .jsx file)
function findDemos() {
  return fs.readdirSync('demos').filter(d => {
    const dir = path.join('demos', d);
    return fs.statSync(dir).isDirectory() && fs.readdirSync(dir).some(f => f.endsWith('.jsx'));
  });
}

if (arg === 'all') {
  // Build landing page
  const distDir = 'dist';
  if (!fs.existsSync(distDir)) fs.mkdirSync(distDir, { recursive: true });
  fs.copyFileSync(path.join('demos', 'index.html'), path.join(distDir, 'index.html'));
  fs.copyFileSync(path.join('demos', 'index.html'), 'index.html');
  console.log('  Built dist/index.html (landing page)');
  console.log('  Built index.html (landing page)');

  // Build each demo
  const demos = findDemos();
  for (const demo of demos) buildDemo(demo);
  console.log('Done: ' + demos.length + ' demo(s) built');
} else {
  buildDemo(arg);
}
