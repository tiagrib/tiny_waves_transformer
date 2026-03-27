// Build script: generates index.html from WaveTransformer.jsx
const fs = require('fs');
const jsx = fs.readFileSync('WaveTransformer.jsx', 'utf8');
// Strip the import/export for inline use
const code = jsx
  .replace(/^import .*$/m, '')
  .replace(/^export default /m, 'const WaveTransformerComponent = ');

const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Wave Transformer</title>
  <style>* { margin: 0; padding: 0; box-sizing: border-box; } body { background: #0a0a0a; }</style>
</head>
<body>
  <div id="root"></div>
  <script src="https://unpkg.com/react@18/umd/react.development.js"><\/script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"><\/script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"><\/script>
  <script type="text/babel">
    const { useState, useRef, useEffect, useCallback, useMemo } = React;

${code}

    ReactDOM.createRoot(document.getElementById('root')).render(
      React.createElement(WaveTransformerComponent)
    );
  <\/script>
</body>
</html>`;

fs.writeFileSync('index.html', html);
console.log('Built index.html');

