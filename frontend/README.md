# Frontend: the browser interface

React + Vite + Three.js. It talks to the backend on port 8000.

```powershell
npm --prefix frontend install        # once
npm --prefix frontend run build      # creates dist/, which the backend serves
npm --prefix frontend run dev        # or: live reload at http://localhost:5173 (backend must be running)
npm --prefix frontend test
```

| Path | What it is |
|---|---|
| `src/components/` | screen parts: camera, captions, 3D hand, speech, ASL Alphabet |
| `src/lib/` | camera, recognition connection, speech and 3D hand logic |
| `public/reference_poses.json` | 3D reference poses and their verification status |
| `dist/`, `node_modules/` | generated, git-ignored, safe to rebuild |
