// Build script: generates standalone HTML from shared core + demo files
// Usage: node build.js [demo-name]  (default: waves)
const fs = require('fs');
const path = require('path');

const demo = process.argv[2] || 'waves';
const demoDir = path.join('demos', demo);

if (!fs.existsSync(demoDir)) {
  console.error('Demo not found: ' + demoDir);
  process.exit(1);
}

// Read shared modules (order matters)
const sharedFiles = ['core.js', 'model.js', 'optimizer.js', 'ui.js'];
const sharedCode = sharedFiles.map(f => {
  let code = fs.readFileSync(path.join('shared', f), 'utf8');
  // Strip Node.js require/exports blocks for browser
  code = code.replace(/if \(typeof require[\s\S]*?\n\}\n?/g, '');
  code = code.replace(/if \(typeof module[^\n]*\n?/g, '');
  return '// --- shared/' + f + ' ---\n' + code.trim();
}).join('\n\n');

// Read demo-specific data.js
let dataCode = '';
const dataPath = path.join(demoDir, 'data.js');
if (fs.existsSync(dataPath)) {
  dataCode = fs.readFileSync(dataPath, 'utf8');
  dataCode = dataCode.replace(/^if \(typeof module.*$/gm, '');
  dataCode = '// --- ' + demo + '/data.js ---\n' + dataCode.trim();
}

// Read demo JSX
const jsxFiles = fs.readdirSync(demoDir).filter(f => f.endsWith('.jsx'));
if (jsxFiles.length === 0) { console.error('No .jsx file found in ' + demoDir); process.exit(1); }
let jsxCode = fs.readFileSync(path.join(demoDir, jsxFiles[0]), 'utf8');
// Strip import/export lines
jsxCode = jsxCode.replace(/^import .*$/gm, '');
jsxCode = jsxCode.replace(/^export default /gm, '');
// Find the component function name
const match = jsxCode.match(/^function\s+(\w+)/m);
const componentName = match ? match[1] : 'App';

// Build title from demo name
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
const outPath = path.join(outDir, 'index.html');
fs.writeFileSync(outPath, html);
console.log('Built ' + outPath);

// Also write to root index.html for the default demo
if (demo === 'waves') {
  fs.writeFileSync('index.html', html);
  console.log('Built index.html (root)');
}
