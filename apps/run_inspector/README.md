# Run Inspector

Web UI for inspecting investigation runs in `results/`.

## Quick start

From the repository root, `agsci inspect start` builds and serves everything (see
`docs/run_inspector.md`). Manually:

```bash
cd apps/run_inspector/frontend
npm install
npm run build

cd /path/to/agentic-scientist
python apps/run_inspector/launcher.py --results-root results --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`.

## Features

### Progressive "build tree" animation

The idea-tree panel has a **▶ build** button (top-right, next to the zoom/fit
controls). Click it to replay the run: starting from the seed program, the tree
grows node-by-node in cycle order. The layout is recomputed at each step from the
nodes revealed so far, so each node's children re-center beneath it and existing
nodes **slide** to their new positions while each new node emerges from its parent
and fades in. The seed-program box stays fixed; a node's children are evenly
spaced (a fixed stride apart), centered directly under it, and wrap to a new row
after every 8 children. When several sub-trees branch from one node (e.g. the
per-worker trees in a parallel run), they are spaced apart by their actual extents
so the trees never overlap. In
lockstep, the performance plot (right panel, "performance" tab) fills in its
points and cumulative-best line as each node appears; the plot's axes are held
fixed (computed from the full run) so they don't jump during playback. The
animation runs to completion and then snaps back to the full view; click **■**
while it is running to stop early. Switching runs resets it.

## Frontend dev

```bash
cd apps/run_inspector/frontend
npm install
npm run dev
```

The frontend proxies `/api` to the backend at `http://127.0.0.1:8765`.
