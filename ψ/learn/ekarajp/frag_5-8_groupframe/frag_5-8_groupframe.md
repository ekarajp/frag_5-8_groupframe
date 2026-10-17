# frag_5-8_groupframe Learning Index

## Source
- **Origin**: ./origin/ (symlink → ghq root, gitignored)
- **GitHub**: https://github.com/ekarajp/frag_5-8_groupframe

## Explorations

### 2026-10-17 1717 (default — 3 agents, Docker-on-Ubuntu focus)
- [[2026-10-17/1717_ARCHITECTURE|Architecture]]
- [[2026-10-17/1717_CODE-SNIPPETS|Code Snippets]]
- [[2026-10-17/1717_QUICK-REFERENCE|Quick Reference]]

**Key insights**:
1. **Single-container, multi-process** — stdlib `ThreadingHTTPServer` dashboard on :8788 (`app/server.py`) → SQLite WAL job queue (`app/engine.py`) → up to N `app/worker.py` subprocesses each importing OpenSeesPy. No broker/DB/external services. The 2,000 grouped-frame models (500 × storeys 5–8) are frozen gzip-JSON OpenSees command scripts under `assets/`, replayed — never generated — at runtime.
2. **Everything is hash-frozen** — Start SHA-256-verifies all 2,616 files in `bundle_manifest.json`; `run.json` freezes code hashes re-verified per job. Docker consequence: build from a pristine checkout, never bind-mount source over `/app`; the only runtime writes are `state/` and `runtime_storage/` (both already volumed in `compose.yml`).
3. **Docker gaps for Ubuntu hardening** — no non-root `USER` (root-owned bind-mount results), no `HEALTHCHECK`, no SIGTERM handler (graceful stop = UI `POST /api/stop` → `stop.flag`, 2h/direction guard; `stop_grace_period: 30s` is only a backstop), `linux/amd64` pin is mandatory (openseespylinux 3.5.1.3 is x86_64-only), `assets/legacy_seeds/` (~643 MB, gitignored but NOT dockerignored) must be prepared before build, and the UI must be opened as `http://127.0.0.1:8788` — the Host-header check 403s `localhost`.
