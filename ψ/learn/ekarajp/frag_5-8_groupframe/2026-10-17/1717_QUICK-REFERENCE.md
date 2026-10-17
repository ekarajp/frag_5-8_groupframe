# QUICK REFERENCE — frag_5-8_groupframe (Component IDA / Fragility, floors 5–8)

Operational cheat sheet for running this app in Docker on Ubuntu.
Entry point: `run.py` → `app/server.py` (stdlib `ThreadingHTTPServer`, HTTP on **127.0.0.1:8788**). Python **3.10 x64**, solver **OpenSeesPy 3.5.1.3** (native engine pinned in `requirements.txt`).

---

## 1. What it does

**Purpose** (sources: `README.md`, `README_TH.md`, `DELIVERY.html`, `component_method.json`):
Incremental Dynamic Analysis (IDA) and seismic fragility computation for **2,000 RC "grouped-frame" building models** — **500 models for each of floor heights 5, 6, 7, 8** ("frag_5-8" = the 5–8-storey program; "groupframe" = the grouped-frame structural model driven by `src/fragility_poc/fishbone_v4_grouped_frame.py` / `fishbone_v5_batched_dynamic.py`).

- **Model inventory**: `assets/catalog.json` (2,000 rows: `stories`, `building_id`, model path, SHA-256, recipe). Per-floor model files under `assets/models/{5,6,7,8}storey/`. Floor 5 uses the *latest mesh16* source snapshot (`src_variants/5storey/`, 5 files differ from 6–8, routing in `source_variants.json`); floors 6–8 use the original delivered sources (`src/`). Selection at runtime: `app/common.py::configure_scientific_source()`.
- **Ground motions**: the original **23 CMS pairs** (`assets/motions.json`, waveforms in `assets/motions/`, 46 component files, T* = 0.2/0.5/1.0/1.5/2.0/3.0 s, set `CMS_ZONE5`, units g). The controller picks the **single nearest CMS family** to `sqrt(T1X·T1Y)` — **3 pairs → 6 component IDA curves** or **4 pairs → 8** (`app/components.py::select_components`). No midpoint blending.
- **Per-building workflow** (`app/engine.py`, `app/worker.py`): one `modal` job per building (gravity + directional modes + component/IM selection), then one `curve` job per component. Each curve is an adaptive hunt (×1.5) + log-bisection search for IO/LS/CP capacities at **MIDR = 0.01 / 0.02 / 0.04**, bracket width ≤ 5% (`app/policy.py::decide`, thresholds in `app/common.py::THRESHOLDS`).
- **IM definition**: `IM = Sa_component(T1_direction, 5%)`; each component (X and Y separately) has its **own** scale factor `SF = IM / unscaled Sa`. No X/Y max combination, no geometric-mean pair IM.
- **Guards** (`app/engine.py::DEFAULT_POLICY`): IM 0.001–20 g, ≤ 100 points/curve, SF ≤ 10,000, 7,200 s/direction. Hitting a guard or a numerical failure stores the point as **needs_review** — it is never counted as CP and never auto-retried.
- **Legacy floor-5 reuse** (`app/legacy_seeds.py`): 192 original mesh16 directional observations (384 files, ~643 MB ZIP) are **not** in Git; they download from GitHub Release `research-data-v1` via `tools/prepare_data.py` (SHA-256 verified, `distribution/legacy_seed_data.json`). Of these, 72 points from 3 buildings match the 500-building catalog and are reused **only** on exact model/candidate/component/GM/T1/hash match; mesh8 history is never mixed in.
- **Outputs**: per-model IDA + fragility HTML/PNG/JSON/CSV + NPZ floor histories and member envelopes (`app/reports.py`), per-height fragility mixture plots, campaign summary, and a SHA-256 `research_inventory.json` export.
- **Dashboard**: local-only web UI (Start / Stop safely / Workers / Export), per-floor and overall progress, live worker telemetry, queue refills immediately when a worker frees (no batch waits). **Opening the dashboard never starts the solver** (`run.py` docstring, `component_method.json: automatic_start:false`).
- **Safety posture**: bundle manifest of 2,616 files (~732 MB) is SHA-256-verified before Start; the frozen experiment plan is hash-locked per run; two controllers on one state are refused via an OS file lock (`app/server.py`, `state/server.lock`).

---

## 2. Installation (every documented method)

### a) Native from source — Linux (README.md "ติดตั้งจากโค้ด: Linux / Windows")
```bash
git clone https://github.com/ekarajp/frag_5-8_groupframe.git
cd frag_5-8_groupframe
python3.10 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/prepare_data.py          # ~643 MB floor-5 seed download, SHA-256 checked
python tools/doctor.py --solver-import  # import/version check only, never analyze()
python app/verify.py                  # solver-free package verification
python run.py                         # dashboard on http://127.0.0.1:8788
```
Ubuntu prereqs: `sudo apt install python3.10-venv libgomp1 libglu1-mesa` (README.md). Headless: `python run.py --no-browser` + `ssh -L 8788:127.0.0.1:8788 user@your-server`.

### b) Native from source — Windows
Same as above but `py -3.10 -m venv .venv` and `.venv\Scripts\activate`; or just double-click **`START_FRAGILITY.bat`** (sets thread env vars, prefers `.venv\Scripts\python.exe`, runs `python -B -X utf8 run.py`).

### c) Linux shell launcher
**`START_FRAGILITY.sh`** — `exec .venv/bin/python -B -X utf8 run.py "$@"` (or system `python3`), passes CLI args through.

### d) Docker (README.md "macOS / Docker Desktop / Linux Docker")
```bash
python3 tools/prepare_data.py    # host Python 3.10+ used ONLY to download data
docker compose up --build
```
Results land in `state/` on the host; cache in `runtime_storage/`. Dashboard bound to loopback only.

### e) Verification scripts (any platform)
- **`VERIFY_PACKAGE.bat`** → `python app/verify.py` (run `python app/verify.py` directly on Linux/Docker).
- **`VERIFY_RESULTS.bat`** → `python app/audit_results.py --deep` (writes `validation/results_audit.json`).
- `distribution/` holds distribution evidence only — `legacy_seed_data.json` (release-asset spec: name/bytes/SHA-256/URL), `original_windows_bundle_manifest.json` (the old Windows bundle), `source_snapshot.json`, `crossplatform_declaration.json` (what changed vs the Windows original: 107 scientific files byte-preserved, 7 launcher/UI files amended, **0 source files edited**).

---

## 3. Existing Docker assets — quoted + assessment

### `Dockerfile` (complete)
```dockerfile
FROM --platform=linux/amd64 python:3.10-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglu1-mesa && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
CMD ["python", "-B", "-X", "utf8", "run.py", "--no-browser", "--bind", "0.0.0.0"]
```

### `compose.yml` (complete)
```yaml
services:
  fragility:
    platform: linux/amd64
    build: .
    ports:
      - "127.0.0.1:8788:8788"
    volumes:
      - ./state:/app/state
      - ./runtime_storage:/app/runtime_storage
    stop_grace_period: 30s
```

### `.dockerignore` (complete)
```
.git
.venv
.downloads
state
runtime
runtime_storage
__pycache__
*.zip
*.part*
```

### What is already right
- **Platform pinning** `linux/amd64` in *both* the Dockerfile and compose.yml — required because `openseespylinux==3.5.1.3` ships x86_64 native wheels; the README explicitly says macOS/ARM must use this image ("x86 emulation on ARM may be slower than native").
- **Loopback-only port map** `127.0.0.1:8788:8788` matches the server's own defenses: `app/server.py::safe_host()` returns **403 unless the `Host` header is exactly `127.0.0.1:8788`**, and every POST additionally needs the session token from `/api/status` plus `Origin: http://127.0.0.1:8788`.
- `--bind 0.0.0.0` in the container + loopback publish is the intended pattern (`--bind` help text: "0.0.0.0 only for a container mapped to host loopback").
- Bind mounts `./state` (all results/queue) and `./runtime_storage` (native cache, forced by `app/common.py` setting `FRAGILITY_POC_RUNTIME_ROOT` to `<repo>/runtime_storage`).
- `stop_grace_period: 30s` gives `docker stop` a drain window.
- `.dockerignore` correctly excludes prior state, caches and ZIPs — the ~643 MB seed ZIP cache (`.downloads`) never enters the build context.
- Thread-pinning env vars (`OMP/OPENBLAS/MKL/NUMEXPR = 1`) match `run.py`/`app/worker.py` — one BLAS thread per native worker process, parallelism comes from N worker processes.

### Missing / worth verifying before a "solid" Ubuntu deployment
1. **No `USER` (non-root) — the biggest operational gap.** The container runs as root, so every file it writes into the `./state` and `./runtime_storage` bind mounts becomes **root-owned on the host**. Fix options: add `USER` in the Dockerfile with a UID/GID matching the host operator, or `user: "${UID:-1000}:${GID:-1000}"` in compose.yml, or plan to `sudo chown -R $USER:$USER state runtime_storage` afterwards. The app needs no privileged operations, so non-root is safe.
2. **No `HEALTHCHECK`.** Good probe: fetch `/api/status` (returns 200 JSON when idle). Caveat: `safe_host()` requires the `Host` header, so a container-local check must send it, e.g.
   `HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8788/api/status',headers={'Host':'127.0.0.1:8788'}),timeout=5)" || exit 1`
   (`curl` is not installed in the slim image; use Python.)
3. **SIGTERM handling / graceful shutdown deserves a live test.** The graceful path (write `stop.flag`, let each worker finish its current IM point, `paused` receipt, resume later) is driven by the **dashboard Stop button** (`Engine.stop()`, `app/engine.py`). `run.py`/`app/server.py` install no SIGTERM handler — `KeyboardInterrupt` is caught only for SIGINT — so `docker compose stop`'s SIGTERM kills the controller without writing `stop.flag`. On reopen, `Engine.__init__` re-checks each `running` job's PID identity (`app/common.py::identity`, psutil PID+exe+creation-time, no stale-PID adoption); dead workers with no valid receipt become `needs_review`, so only the in-flight point is lost, but it needs manual review rather than auto-resume. Runbook consequence: **always press "Stop safely" and wait for active = 0 before `docker compose down`** (README.md says exactly this).
4. **Image size**: `COPY . ./` bakes in the whole ~732 MB asset tree (2,000 models + 46 waveforms + reference reports). Expect a multi-GB image (numpy/scipy/pandas/matplotlib + OpenSeesPy on top). Fine for a research box; don't expect a slim image without splitting assets onto a volume.
5. **arm64 story**: no multi-arch build. On ARM hosts (Apple Silicon, ARM servers) the amd64 pin means QEMU/Rosetta emulation — functional (this is the documented macOS route) but slower; for a native-ARM Ubuntu host there is currently **no supported solver** (README: "not all OS/architectures supported automatically").
6. **Port is effectively fixed at 8788.** Because of the `Host == 127.0.0.1:8788` check, remapping to e.g. `127.0.0.1:9000:8788` makes every request 403 (`Local dashboard Host required`). To use another host port you would have to edit `app/server.py` (`safe_host`, token Origin check, and `dashboard.json` writer all hard-code the port).
7. Minor: no `restart` policy in compose.yml (reasonable — auto-restart would violate the "no automatic analyses" contract; leave it off); no `init: true` (workers are direct children; PID-1 zombie reaping is handled by psutil zombie detection, but `init: true` is cheap insurance); no memory/CPU limits (workers = `min(7, cpu_count)` by default; cap via `deploy.resources` or set workers low in the UI).

---

## 4. Key features with examples

- **Analysis modes** (two job kinds, `app/engine.py`, `app/worker.py`):
  - `modal` — gravity equilibrium (error must be ≤ 1%) + directional modes + CMS family/component selection per building; emits the 6–8 `curve` jobs (`worker.py::modal`, `components.py::select_components`).
  - `curve` — per-component IDA: adaptive hunt ×1.5 from 0.05 g, log-bisection to ≤ 5% brackets on IO/LS/CP; each point is one NLTHA at a single scale factor with full history capture and member localForce/basicDeformation envelopes streamed from native recorder files (`worker.py::curve`). Queue is work-conserving (0.15 s refill loop, curve jobs before modal).
  - **Legacy reuse** — `app/legacy_seeds.py::import_seeds` injects hash-verified original floor-5 directional observations as IDA points before any new analysis; signed failures are never retried (`validation/legacy_reuse_validation.json`: accepted example 3, idempotent, cross-height import rejected).
- **Results / plots** (all under `state/<run-*/>`):
  - Per model `reports/<stories>-<building_id>/`: `index.html`, `ida_fragility.png` (IDA curves + fragility CDFs), `ida_points.csv`, `component_capacities.csv`, `results.json` (incl. lognormal fit θ, β, bootstrap 95% CI resampled by **parent pair**, seed 68001500 — `reports.py::fit_complete`), `manifest.json`. Final fit is **withheld** unless all 6/8 components are complete and certified (`reports.py::model_report`).
  - Campaign level: `campaign_summary.json`, `fragility_summary.csv`, `height-{5..8}-fragility.png` (equal-building-weight mixture; states `incomplete_final_mixture_withheld` until a height's 500 are done), `index.html`, `research_inventory.json` (SHA-256 of every retained file), `job_ledger.json`, `queue_snapshot.sqlite` (`engine.py::finalize`).
- **Verification workflow** (all solver-free, `analysis_calls: 0`):
  - `python app/verify.py` (= `VERIFY_PACKAGE.bat`): replays X/Y construction commands for all 2,000 models (4,000 replays) against `assets/models/*/manifest.json`, checks 23 CMS pairs, writes `validation/package_verification.json` (recorded result: 2,616 files, `sha256 f8990388…`).
  - `python app/audit_results.py --deep` (= `VERIFY_RESULTS.bat`): re-derives frozen `run_hash`, verifies every point's `file_hashes`, optionally re-validates NPZ histories vs `max_midr`; exits 1 on any error; writes `validation/results_audit.json`.
  - `python -m unittest discover -s tests -p "test_*.py"` — 13 tests (policy, queue with `tests/fake_worker.py`, source routing, dashboard HTTP idle-start + 403 on tokenless Start; log in `validation/unit_queue_tests.log`).
  - `python tests/portability_no_analysis.py` (source routing + Thai-unicode/C-drive relocation), `python tests/seed_probe.py`, `python tools/doctor.py --solver-import`.
  - CI: `.github/workflows/portability.yml` runs the whole chain on `ubuntu-22.04` + `windows-latest`.
  - README caveat: passing these checks is **not** full-population qualification — the 2,000-building campaign must still run on the target machine.

---

## 5. Configuration options (every knob)

### CLI flags
| Command | Flags |
|---|---|
| `python run.py` → `app/server.py` | `--port` (default 8788), `--no-browser`, `--state <dir>` (default `ROOT/state`), `--bind {127.0.0.1,0.0.0.0}` (default loopback; `0.0.0.0` only inside a container) |
| `app/worker.py` (launched by engine) | `--job <job.json>`, `--run <run.json>` |
| `app/audit_results.py` | `--run <run-dir>` (default from `state/current.json`), `--deep` (NPZ re-validation) |
| `tools/doctor.py` | `--solver-import` (also imports the native engine and prints `ops.version()`) |
| `tools/prepare_data.py` | `--archive /path/to/floor5-legacy-seeds-v1.zip` (offline install instead of download) |

### Environment variables
- Forced to `1` by `run.py`, `worker.py`, and the Dockerfile: `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`.
- `MPLBACKEND=Agg` (set in `run.py` via `setdefault` and the Dockerfile).
- Set by `app/common.py` at import (do not override casually): `FRAGILITY_POC_RUNTIME_ROOT=<repo>/runtime_storage`, `FRAGILITY_ALLOW_SYSTEM_DRIVE=1`.
- Set per worker by `configure_scientific_source()`: `COMPONENT_SCIENTIFIC_VARIANT` (`5storey-original` or `6-8-original`) — recorded in every run's environment snapshot.

### JSON configuration / state files
- `state/settings.json` — only knob: `{"workers": N}` (written by the UI/API; reloaded on restart; clamped to 1..cpu_count).
- `state/current.json` — points at the active `run-*` directory.
- `state/<run-*/run.json` — **frozen** experiment plan: the policy dict (below), bundle/catalog/motions SHA-256s, target counts, environment snapshot, `code_hashes` of every app/src file; `run_hash` digest is re-verified on every open and by every worker. Changing inputs silently is refused.
- Policy defaults (`app/engine.py::DEFAULT_POLICY`, copied into `run.json` only for *new* runs): `initial_im_g .05`, `minimum_im_g .001`, `maximum_im_g 20`, `hunt_multiplier 1.5`, `bracket_tolerance .05`, `maximum_points_per_curve 100`, `maximum_scale_factor 10000`, `runtime_s_per_direction 7200`, `damping_ratio .05`, `numerical_failure_policy retain/needs_review/no unchanged retry`.
- Read-only inputs: `assets/catalog.json` (2,000 models), `assets/motions.json` (23 CMS pairs), `assets/legacy_seed_index.json`, `bundle_manifest.json` (2,616 file hashes), `component_method.json`, `source_variants.json` (5→`src_variants/5storey`, 6–8→`src`), `distribution/legacy_seed_data.json` (release URL + SHA-256 of the 643 MB seed ZIP).
- `state/dashboard.json` — informational: dashboard URL, PID, `production_autostart: false`.

**Important**: the policy is frozen per run. To change it you start a **new run directory** (delete/move `state/current.json`); `Engine.start()` additionally refuses to run if `bundle_manifest.json`'s hash no longer matches the frozen plan.

---

## 6. Usage patterns

### A. Build & run with Docker on Ubuntu
```bash
cd frag_5-8_groupframe
python3 tools/prepare_data.py        # BEFORE build: seeds are baked into the image (assets/ is not dockerignored)
docker compose build
docker compose up -d
docker compose logs -f fragility     # expect: "Dashboard ready: http://127.0.0.1:8788 — waiting for Start"
```
Then open **http://127.0.0.1:8788** in the host browser. Because the publish is loopback-only, remote access = SSH tunnel (`ssh -L 8788:127.0.0.1:8788 user@ubuntu-host`), never a public port.

### B. HTTP endpoints (`app/server.py`)
| Method/Path | Notes |
|---|---|
| `GET /` | serves `app/dashboard.html` (Thai UI, "Component IDA Lab / 5–8 floors") |
| `GET /api/status` | full engine snapshot + session `token` + export state (polled every 2 s by the UI) |
| `GET /api/reports` | list of finished per-model report pages |
| `GET /results/<path>` | files from the current run dir (IDA PNG/HTML/CSV, campaign `index.html`, inventory) — path-confined |
| `GET /reference/<path>` | original Development reports from `assets/models/...` — path-confined |
| `POST /api/start` | body `{"workers": N}`; runs one-time preflight SHA-256 verification of all 2,616 files, then runs continuously |
| `POST /api/stop` | safe stop: stop dispatching, let each worker finish its current IM point, then pause |
| `POST /api/workers` | body `{"workers": N}`; live resize — running jobs are never killed |
| `POST /api/export` | refused while workers are active; freezes `queue_snapshot.sqlite`, `job_ledger.json`, campaign reports, `research_inventory.json` |
All POSTs require the `X-Runner-Token` header (from `/api/status`) and matching `Origin`; all requests require `Host: 127.0.0.1:8788`; request body capped at 4,096 bytes. A second controller on the same state exits with *"This portable campaign already has a dashboard/controller"* (flock on `state/server.lock`).

### C. Typical operator workflow in the browser
1. Open dashboard → status shows "Ready — no analyses started", 2,000 jobs pending.
2. Press **Start / Resume** → preflight "ตรวจ SHA-256 inputs…" progress, then dispatch begins with `workers = 7` (default `min(7, cpu_count)`; adjust and **Apply workers** any time).
3. Watch: per-floor tiles (processed/certified/needs-review), live worker table (axis · IM · record %), certified vs processed percentages (processed includes needs_review by design).
4. **Stop safely** → wait until *active = 0* (each worker pauses at the next IM-point boundary with a `paused` receipt).
5. **Export research inventory** → wait for `export: complete` → open **/results/index.html** (campaign report link appears in the UI when complete).
6. Post-run: `python app/audit_results.py --deep` inside the repo (or container) → `validation/results_audit.json`.
7. Results live entirely on the host under `./state/<run-*/>`; copy-back procedure is documented in `RETURN_RESULTS_TH.md` (take the whole `state/` folder incl. `queue.sqlite-wal/-shm`; never only the summary plots; never delete needs_review cases).

---

## 7. Ubuntu + Docker runbook

```bash
# 0) prerequisites (Ubuntu 22.04/24.04, x86_64)
docker --version && docker compose version        # compose v2 plugin
python3 --version                                  # 3.10+ needed only for the data download step

# 1) fetch the ~643 MB floor-5 legacy seeds (SHA-256-verified into assets/legacy_seeds/)
cd /path/to/frag_5-8_groupframe
python3 tools/prepare_data.py                      # offline alt: --archive /path/floor5-legacy-seeds-v1.zip

# 2) build & start (amd64 pin applies automatically)
docker compose build
docker compose up -d
docker compose logs -f fragility                   # Ctrl+C to stop following; container keeps running
ss -ltn | grep 8788                                # confirm loopback-only listener

# 3) use it
xdg-open http://127.0.0.1:8788                     # or SSH tunnel from your laptop
#    → Start / Resume (choose workers) → wait → Stop safely → active=0 → Export research inventory

# 4) where results land (host bind mounts)
ls state/                                          # current.json, settings.json, server.lock
ls state/run-*/                                    # run.json queue.sqlite jobs/ reports/ events.jsonl
ls state/run-*/reports/                            # per-model report folders
xdg-open state/run-*/index.html                    # campaign report (after Export)

# 5) verify without the solver
python3 app/verify.py                              # package check -> validation/package_verification.json
python3 app/audit_results.py --deep                # results check -> validation/results_audit.json

# 6) stop
#    dashboard: Stop safely, wait active = 0, Export complete — THEN:
docker compose stop                                # 30s stop_grace_period
docker compose down                                # containers removed; ./state and ./runtime_storage kept
```

**Graceful shutdown**: the 30 s `stop_grace_period` only buys time — the *reliable* drain is the dashboard's **Stop safely** button (writes `stop.flag`; each worker finishes its current IM point and stores a `paused` receipt; Resume later picks up from committed, hash-verified points only). If the container is killed hard, interrupted jobs surface as `needs_review` on the next open (no receipt) — that is by design, not corruption; committed points are never re-run.

**Common pitfalls**
- **Root-owned results**: with the stock Dockerfile the container runs as root, so `state/` and `runtime_storage/` fill with root-owned files on the host. Either add a matching `USER`/`user:` to the compose service before first run, or reclaim afterwards with `sudo chown -R $USER:$USER state runtime_storage`.
- **Port conflicts**: something already on 8788 → `docker compose up` fails to bind. Free the port; do **not** remap to a different host port — the server 403s any `Host` other than `127.0.0.1:8788` (`app/server.py::safe_host`), so `127.0.0.1:9000:8788` yields a dashboard that loads nothing.
- **Two controllers**: never start a second copy (`run.py` outside Docker + the container) against the same `state/` — flock makes the second exit, and concurrent SQLite/flag writes are exactly what the design forbids (README: "ห้ามเปิดสอง controller ต่อ state เดียวกัน").
- **arm64 host (Apple Silicon / ARM server)**: `platform: linux/amd64` forces QEMU/Rosetta emulation — works (it is the documented macOS route) but slower; expect longer wall-clock per NLTHA. There is no native ARM solver wheel (`requirements.txt` header: "macOS and other architectures: use the linux/amd64 Docker image").
- **Skipped `prepare_data.py`**: the Start preflight verifies all 2,616 manifest files **including the 385 floor-5 seed files**, so Start fails without them; if you ran `prepare_data.py` *after* building, the seeds are on the host but not in the image — rebuild (`docker compose build`) or mount `./assets/legacy_seeds:/app/assets/legacy_seeds:ro`.
- **Disk space**: the controller auto-stops when the volume holding `state/` drops below 10 GiB free (`Engine.minimum_free_gib`; reason string: "Disk reserve reached… move the whole stopped package to a larger disk, then resume"). Each IM point stores full floor-history NPZs — budget tens of GB for a full campaign.
- **Moving hosts**: Stop → active = 0 → close controller → copy the **whole `state/` folder** (incl. `-wal`/`-shm`) with the same source/assets versions; verify environment + hashes before Resume; no bit-identical cross-OS floating-point claims (README.md; run environment is recorded in `run.json`).

---

### File map (fast lookup)
| Area | Path |
|---|---|
| Entry / dashboard server | `run.py`, `app/server.py`, `app/dashboard.html` |
| Queue/controller | `app/engine.py` (defaults, start/stop, finalize) |
| Native worker / IDA | `app/worker.py`, `app/policy.py`, `app/components.py`, `app/legacy_seeds.py` |
| Reports / audit | `app/reports.py`, `app/verify.py`, `app/audit_results.py` |
| Common utils / hashes | `app/common.py` |
| Tools | `tools/prepare_data.py`, `tools/doctor.py` |
| Tests | `tests/test_runner.py`, `tests/test_dashboard_portable.py`, `tests/test_portable_platform.py`, `tests/test_components.py`, `tests/test_all_heights.py`, `tests/portability_no_analysis.py`, `tests/seed_probe.py` |
| Docs | `README.md`, `README_TH.md`, `RETURN_RESULTS_TH.md`, `DELIVERY.html` |
| Docker | `Dockerfile`, `compose.yml`, `.dockerignore` |
| Provenance | `bundle_manifest.json`, `component_method.json`, `source_variants.json`, `distribution/*`, `validation/*` |
