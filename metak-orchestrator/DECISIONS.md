# Decisions Log

Decisions made autonomously during unattended implementation.

## D1: Development Strategy — Build Standalone, Then Merge
**Date:** 2026-03-27
**Decision:** Build the model logic as standalone JS files (`src/model.js`) testable with Node.js, then merge everything into a single `WaveTransformer.jsx` at the end.
**Rationale:** The spec requires a single JSX file, but developing and testing inline is impractical. Standalone files allow running automated tests with Node.js during development.

## D2: Analytical Gradients as Primary Training Method
**Date:** 2026-03-27
**Decision:** Implement analytical backprop verified against numerical gradients. Use analytical for training (spec recommends this for performance).
**Rationale:** Numerical gradients (~4,320 forward passes × 24 batch) would be too slow for the 50ms training tick. Analytical backprop is needed, but must be verified first.

## D3: Test Infrastructure
**Date:** 2026-03-27
**Decision:** Use plain Node.js assert-based tests (no test framework dependency). Tests live in `tests/` directory.
**Rationale:** Keep dependencies minimal. The project has no package.json and the final deliverable is a single JSX file.
