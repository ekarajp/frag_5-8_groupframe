---
pattern: "Learned frag_5-8_groupframe: single-container multi-process OpenSees fragility app, hash-frozen bundle (build pristine, volume only state/+runtime_storage/), Docker gaps = non-root user, healthcheck, SIGTERM drain, amd64 pin, pre-build seeds, 127.0.0.1-only Host"
date: 2026-10-17
source: learn: ekarajp/frag_5-8_groupframe
concepts: ["learn", "codebase", "docker", "ubuntu", "opensees", "fragility-analysis", "seismic"]
---

# Learned frag_5-8_groupframe

Goal of the session: dockerize the app to run on Ubuntu. It already ships a minimal `Dockerfile` + `compose.yml` (python:3.10-slim-bookworm, `linux/amd64` pin, port 127.0.0.1:8788, volumes `./state` + `./runtime_storage`).

1. **Runtime shape**: `run.py` → `app/server.py` (stdlib ThreadingHTTPServer, port 8788, flock single-instance, Host-header + token auth on every request) → `app/engine.py` (SQLite WAL queue) → ≤N `app/worker.py` OpenSees subprocesses. The 2,000 grouped-frame models are frozen replayable OpenSees command scripts in `assets/` — nothing is generated at runtime.

2. **Hash-frozen integrity**: Start verifies `bundle_manifest.json` (2,616 files, SHA-256); `run.json` code hashes are re-verified per job. So: never mount host source over `/app`; only `state/` and `runtime_storage/` are runtime writes (already volumed). `assets/legacy_seeds/` (~643 MB) is gitignored but NOT dockerignored and must exist before Start — run `tools/prepare_data.py` before building the image.

3. **Hardening gaps found**: no non-root USER (bind-mount results become root-owned), no HEALTHCHECK, no SIGTERM handler (graceful stop is app-level `stop.flag`; ungraceful kill is survivable via receipts/point-hash resume but yields `needs_review`), amd64 pin mandatory (openseespylinux 3.5.1.3 x86_64-only), and the dashboard only accepts Host `127.0.0.1:8788` — `http://localhost:8788` gets 403.

Full docs: ψ/learn/ekarajp/frag_5-8_groupframe/2026-10-17/1717_{ARCHITECTURE,CODE-SNIPPETS,QUICK-REFERENCE}.md
