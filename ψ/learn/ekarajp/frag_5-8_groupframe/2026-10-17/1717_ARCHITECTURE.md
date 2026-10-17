# ARCHITECTURE — frag_5-8_groupframe (Component IDA / Fragility, RC Grouped Frames 5–8 storeys)

Prepared for dockerization on Ubuntu. All paths are relative to the repo root.
Sources: file inspection on 2026-10-17; commit `3f27608` ("Verify idle dashboard startup and reject unauthenticated start").

## 1. What this is

A local-only research application that runs **Incremental Dynamic Analysis (IDA)** of **2,000 reduced RC "grouped frame" building models** (500 each at 5, 6, 7, 8 storeys) under **23 CMS ground-motion pairs**, using **OpenSeesPy 3.5.1.3** as the native FEA engine, and produces per-model component fragility fits plus a campaign-level export. A single-file HTTP dashboard (`app/server.py`, port 8788) controls the campaign; **opening the dashboard never starts analyses — Start is an explicit, authenticated action** (`run.py` docstring; `app/server.py`; `component_method.json` declares `"automatic_start": false`).

The defining engineering constraint: this is a *frozen scientific experiment*. Every input file, every source file, every committed result carries a SHA-256, verified at Start (preflight) and on every job resume (`bundle_manifest.json`, `app/common.py:verify_assets`, `app/worker.py`). Dockerization must not perturb those hashes.

## 2. Directory structure

| Path | Contents | Notes |
|---|---|---|
| `app/` | The portable application layer (11 files): HTTP server, queue engine, native worker, policy, component selection, reports, verification, audit, dashboard UI | No OpenSees import in the dashboard process (`app/common.py` module docstring) |
| `src/` | The "6–8 storey" scientific source snapshot: Python package `fragility_poc` (49 modules, incl. `structural.py` 126 KB, `ida.py` 158 KB, `ground_motion.py` 69 KB, `fishbone_v4_grouped_frame.py`, `fishbone_v5_batched_dynamic.py`) plus `rc_fragility_poc.egg-info` (historical package metadata) | Put on `sys.path` at runtime, **not pip-installed** (no `pip install -e .` in `Dockerfile`) |
| `src_variants/5storey/` | The "5-storey" scientific snapshot: a second copy of `fragility_poc` + its own `egg-info` | Differs from `src/` in exactly 5 files per `source_variants.json`: `catalog.py`, `config.py`, `io_utils.py`, `runtime_storage.py`, `structural.py` (`"source_files_edited": 0` — both are byte-frozen originals) |
| `assets/` | Immutable research inputs (2,480 of the 2,616 manifest files): `catalog.json` (2,000 model rows), `models/{5,6,7,8}storey/` (500 `B-*.json.gz` construction definitions per height + manifest/recipe metadata), `motions/` (46 CMS component `.txt` files), `motions.json` (23 pair descriptors), `grouped_model_loader.py`, `legacy_seed_index.json`, `provenance/` (preregistrations/amendments), `reuse_summary.json` | All hash-verified; `assets/legacy_seeds/` (~643 MB) is **absent from Git** and downloaded by `tools/prepare_data.py` |
| `distribution/` | Provenance/evidence of the original Windows-only distribution: `original_windows_bundle_manifest.json` (3.75 MB), `source_snapshot.json`, `crossplatform_declaration.json` (the OS-portability amendment, 7 changed files, `"population_analyses_run": 0`), `legacy_seed_data.json` (GitHub Release URL + 384-file manifest) | Records, not runtime inputs |
| `tools/` | Two operator CLIs: `prepare_data.py` (download/verify legacy seeds), `doctor.py` (environment + optional solver-import probe) | Both solver-safe (`doctor.py` "without analyze()") |
| `tests/` | Unit/queue/HTTP/portability tests, plus two solver-free probes (`portability_no_analysis.py`, `seed_probe.py`, `source_route_probe.py`) and `fake_worker.py` (deterministic stand-in for `app/worker.py`) | Queue tests use the fake worker, never OpenSees |
| `validation/` | Committed validation records (`dashboard_http.json`, `package_verification.json`, `source_routing_and_relocation.json`, `prior-native-validation/{5storey,6-8storey}/`, …). `verify.py` and `audit_results.py` **write** `package_verification.json` / `results_audit.json` here at runtime | Write target inside the source tree (see §6) |
| Root files | `run.py`, `START_FRAGILITY.{sh,bat}`, `VERIFY_{PACKAGE,RESULTS}.bat`, `requirements.txt`, `Dockerfile`, `compose.yml`, `.dockerignore`, `bundle_manifest.json` (2,616-file manifest, 519 KB), `component_method.json`, `source_variants.json`, READMEs (TH), `DELIVERY.html` | |
| `.github/workflows/portability.yml` | CI: ubuntu-22.04 + windows-latest matrix; installs `libgomp1 libglu1-mesa` on Linux, runs the full no-analysis verification suite | The Linux CI recipe is effectively the Dockerfile recipe |

**Organization philosophy**: a strict three-ring split — (1) `app/` orchestration that never imports OpenSees, (2) frozen scientific snapshots (`src/`, `src_variants/`) that never know about the dashboard, (3) hash-pinned immutable assets. Everything is checksummed by `bundle_manifest.json`; `app/engine.py:code_hashes` freezes the hashes of `app/`, `src/`, `src_variants/` and `assets/grouped_model_loader.py` into each run plan, and `app/worker.py` re-verifies them per job ("Frozen scientific/application source changed").

## 3. Entry points (all of them)

| Entry point | What it does |
|---|---|
| `run.py` | Cross-platform main entry. Pins `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`, forces `MPLBACKEND=Agg`, puts `app/` on `sys.path`, then `runpy`s `app/server.py`. CLI flags (`--port` 8788, `--no-browser`, `--state`, `--bind {127.0.0.1,0.0.0.0}`) are parsed by `app/server.py`. |
| `app/server.py` | The dashboard/controller. `ThreadingHTTPServer` on the chosen port; single-instance enforced by an OS lock on `state/server.lock` (`fcntl` on POSIX, `msvcrt` on Windows); writes `state/dashboard.json`; generates a per-boot `X-Runner-Token`; starts `Engine`. Endpoints: GET `/api/status`, `/api/reports`, `/results/*` (serves `run_dir`), `/reference/*` (serves `assets/`), `/`; POST `/api/start`, `/api/stop`, `/api/workers`, `/api/export`. |
| `app/worker.py` | The native analysis process — launched by `Engine.launch` as `python -B -X utf8 app/worker.py --job <run>/jobs/<id>/job.json --run <run>/run.json` (one subprocess per job, `cwd=ROOT`). Handles both job kinds (`modal`, `curve`), imports OpenSees, writes `receipt.json`/`error.json`. |
| `START_FRAGILITY.sh` / `START_FRAGILITY.bat` | Operator launchers: prefer `.venv` python, exec `python -B -X utf8 run.py`. The `.bat` also sets the four thread-pin env vars. |
| `VERIFY_PACKAGE.bat` | Runs `app/verify.py` — read-only package verification: all 2,616 manifest SHA-256s, 2,000 catalog rows × X/Y construction replays (4,000 total) with a recording stub, 23 CMS pairs, **0 solver imports**; writes `validation/package_verification.json`. |
| `VERIFY_RESULTS.bat` | Runs `app/audit_results.py --deep` — audits a finished run: run-hash integrity, every committed point's `file_hashes`, optional NPZ re-validation, `research_inventory.json` cross-check; writes `validation/results_audit.json`. |
| `tools/prepare_data.py` | Downloads `floor5-legacy-seeds-v1.zip` (~643 MB) from the GitHub Release `research-data-v1` (URL in `distribution/legacy_seed_data.json`) or `--archive <zip>` offline; verifies archive + per-file SHA-256; extracts to `assets/legacy_seeds/`; caches in `.downloads/`. No solver. **Must run before Start** (`bundle_manifest.json:missing_research_data_instruction`; legacy seeds are needed because floor-5 curve jobs import prior observations via `app/legacy_seeds.py`). |
| `tools/doctor.py` | Asserts Python 3.10 + Windows/Linux, checks package versions (`openseespy` and the platform-native wheel must both be 3.5.1.3), optional `--solver-import` imports `openseespy.opensees` and reports `ops.version()` — never calls `analyze()`. |
| `tests/…`, `.github/workflows/portability.yml` | CI entry points (`unittest discover`, the three probes, `app/verify.py`). |
| (historical) `src/rc_fragility_poc.egg-info/entry_points.txt` | `fragility-poc = fragility_poc.cli:main` console script from the original package. Not installed in Docker (only `requirements.txt` is pip-installed); the portable app reaches `fragility_poc` via `sys.path.insert` in `app/common.py`. |

## 4. Core abstractions and relationships

### 4.1 Process / component graph

```
run.py
  └─ app/server.py  (HTTP :8788, ThreadingHTTPServer, single instance via state/server.lock)
       └─ Engine (app/engine.py)   [1 controller process]
            ├─ daemon loop thread        — dispatch, reap, refill, disk guard
            ├─ ThreadPoolExecutor(1)     — per-model report I/O + final export
            ├─ sqlite3 queue.sqlite      — jobs/model_reports tables, WAL mode
            └─ subprocess(es): app/worker.py --job … --run …   [N native workers]
                 ├─ app/common.configure_scientific_source(stories) → picks src/ vs src_variants/5storey/
                 ├─ assets/grouped_model_loader.py  — replays whitelisted construction commands into OpenSees
                 ├─ fragility_poc.* (structural, fishbone_v4_grouped_frame, fishbone_v5_batched_dynamic,
                 │                    ground_motion, fishbone_ida) + openseespy native engine
                 ├─ app/policy.decide        — hunt + log-bisection IM selection
                 ├─ app/components.select_components — CMS family/IM selection (modal job)
                 └─ app/legacy_seeds.import_seeds    — floor-5 prior-observation reuse (curve job)
```

### 4.2 The job model (how the "2,000 models" become work)

- `assets/catalog.json` holds 2,000 rows — `{building_id, model_hash, path, sha256, bytes, stories, recipe, candidate_hash}` — exactly 500 per storey (verified in `app/verify.py`; `assets/models/<n>storey/manifest.json` mirrors each height).
- On first open, `Engine.__init__` seeds **one `modal` job per model** (2,000 jobs) into `queue.sqlite` (`app/engine.py` "CREATE TABLE IF NOT EXISTS jobs …", seed loop).
- A `modal` worker builds the model in X and Y (`assets/grouped_model_loader.build_model` — gzip-JSON `portable-grouped-construction-v1` command scripts replayed into a fresh `ops` domain; command whitelist in `CONSTRUCTION_COMMANDS`), runs gravity (`run_grouped_frame_gravity`, error must be ≤1%), extracts directional modes, then `app/components.select_components` picks the **single nearest CMS family** (3 or 4 pairs, chosen by `|√(T1x·T1y) − T*|` from `assets/motions.json`) and computes each component's unscaled `Sa(T1_direction, 5%)`.
- On a valid modal receipt, `Engine.finish` fans out **6 or 8 `curve` jobs** per model. So the queue holds 2,000 modal + up to 16,000 curve jobs.
- A `curve` worker replays the exact same construction commands (`dyn.build_grouped_frame_model = build` monkey-patch, `app/worker.py:82`), then loops: `app/policy.decide` (hunt ×1.5 from 0.05 g, log-bisection to ≤5% bracket, IO/LS/CP = .01/.02/.04 via `fragility_poc.fishbone_ida.estimate_limit_state_capacities`) → run one direction of `fishbone_v5_batched_dynamic._run_direction_batched` at the chosen IM → validate the history arrays → commit the point atomically with hashes. Guards (max IM 20 g, 100 points, SF 10,000, 7,200 s/direction) terminate to `needs_review`; numerical failures are retained, never treated as CP crossings and never auto-retried (`app/policy.py`).
- `curve` jobs are dispatched with priority over `modal` (`ORDER BY CASE kind WHEN 'curve' THEN 0 ELSE 1 END`, dispatch loop) and the loop refills free slots immediately (0.15 s tick) — "work-conserving queue. No batches" (`app/engine.py` docstring).
- Floor-5 curves first call `app/legacy_seeds.import_seeds`: pre-existing directional observations from `assets/legacy_seed_index.json` (64 keys → metadata+history under `assets/legacy_seeds/`) are imported as points **only** if model/candidate/component/GM/period/hashes all match — otherwise they're excluded with a reason; signed original failures are refused outright.

### 4.3 Trust model (why the code is shaped this way)

- **Freeze**: `run.json` (the plan) is self-hashed (`run_hash = digest(run)`); reopening a run re-verifies it ("Frozen experiment plan changed"). It records manifest/catalog/motion SHA-256s, environment (`app/common.environment`: python, platform, executable, package versions, cpu count, `COMPONENT_SCIENTIFIC_VARIANT`), and `code_hashes` of all application+scientific sources.
- **Receipts**: a worker is believed only via `receipt.json` whose `job_id`/`run_hash` match, exit code is 0 (or adopted live), and whose `result_sha256`/`curve_sha256` match on-disk files. A paused receipt (`stop.flag` seen between points) re-queues the job after renaming the old receipt aside.
- **Resume**: only complete, hash-verified `point-*/component.json` folders are reused; an uncommitted point dir or scratch dir aborts with "audit required, automatic retry refused" (`app/worker.py` resume block).
- **Identity**: worker PIDs are tracked as `{pid, exe, created}` — `psutil` on Linux, raw `ctypes`/kernel32 on Windows (`app/common.identity`). A reopened controller adopts genuinely-live orphans and never duplicates them (queue test `test_reopen_adopts_live_workers_without_duplicate`).
- **Single writer**: `state/server.lock` (OS lock) guarantees one controller per state dir; `compose.yml` mounts one state dir.

### 4.4 Scientific source variant routing

`app/common.configure_scientific_source(stories)` is called by every worker **before** importing any `fragility_poc` module: stories 5 → `sys.path.insert(0, ROOT/'src_variants/5storey')`; 6/7/8 → `ROOT/'src'`. For 5-storey it additionally shims the old `runtime_storage.configure_runtime_storage` (which inferred its root from source depth and rejected `C:`) so it routes through the portable helper with an explicit `ROOT/runtime_storage` (`app/common.py` lines 143–159; rationale in `source_variants.json` and `distribution/crossplatform_declaration.json` — 5-storey models use the latest "mesh16" recipe, 6–8 use the original delivery models; `README.md` §วิธีคำนวณ).

### 4.5 Reporting

- Per model (once all its jobs are terminal, scheduled on the single-thread pool so it never starves native dispatch): `app/reports.model_report` → `state/<run>/reports/<stories>-<building_id>/{index.html, ida_fragility.png, ida_points.csv, component_capacities.csv, results.json, manifest.json}`. The lognormal fragility fit (`fit_complete`) is produced **only** when all 6 or 8 component curves are complete and certified; uncertainty is cluster-bootstrapped by parent CMS pair (components of one pair are not independent events).
- Campaign export (POST `/api/export`, or automatically when the queue drains): `Engine.finalize` → SQLite snapshot `queue_snapshot.sqlite` + `job_ledger.json` + `app/reports.campaign_export` → `fragility_summary.csv`, `campaign_summary.json`, `height-{5..8}-fragility.png`, `index.html`, and `research_inventory.json` (SHA-256 inventory of every file in the run dir).

## 5. Dependencies

### 5.1 Python (from `requirements.txt`, all pinned)

```
numpy==1.26.4  scipy==1.15.3  matplotlib==3.10.9  pandas==2.3.3
joblib==1.5.3  psutil==7.0.0  openseespy==3.5.1.3
openseespywin==3.5.1.3; sys_platform == "win32"
openseespylinux==3.5.1.3; sys_platform == "linux"
```

- Header comment: "Python 3.10 x64; native Linux x86_64 / Windows x64. macOS and other architectures: use the linux/amd64 Docker image." (`tools/doctor.py` enforces both the 3.10 and the Windows/Linux parts.)
- **openseespy vs openseespylinux/openseespywin**: the pure-Python `openseespy` package is the API; the platform-marked wheel ships the actual compiled OpenSees interpreter. Both must be 3.5.1.3 (`tools/doctor.py` lines 13–16) — this pins the physics binary to the version used for the original research. Only the matching extra installs on each OS (env markers). Code references: `app/common.environment()` lists all three versions into each run record; `src/fragility_poc/structural.py:_ops()` is the single `import openseespy.opensees` site, then `native_paths.install_native_paths(ops)` rewrites recorder paths to ASCII-relative ones.
- `psutil` is functional on Linux: `app/common.identity` uses it for crash-safe worker identity (the Windows branch uses `ctypes` instead).
- `matplotlib` is used headless only (`MPLBACKEND=Agg` in `run.py`, the Dockerfile ENV, and `matplotlib.use('Agg')` inside `app/reports.py`).
- `pandas`/`joblib` (and `scikit-learn`, present in the historical `src/rc_fragility_poc.egg-info/requires.txt` but **not** in `requirements.txt`) are not imported by the runtime hot path (`structural.py`, `ground_motion.py`, `fishbone_v4_grouped_frame.py`, `fishbone_v5_batched_dynamic.py`, `fishbone_ida.py`, `app/*`) — they remain pinned for the wider frozen package. `numpy` and `scipy` (`scipy.special.ndtr` in the fragility fit, spectral ordinates in `ground_motion.py`) are load-bearing.

### 5.2 System libraries (why the Dockerfile installs them)

`Dockerfile` line 3: `apt-get install -y --no-install-recommends libgomp1 libglu1-mesa` on `python:3.10-slim-bookworm`.

- **`libgomp1`** — GNU OpenMP runtime (`libgomp.so.1`). The OpenSees native binary (and OpenBLAS underneath numpy/scipy) is OpenMP-linked; slim Debian images ship without it, and `import openseespy.opensees` fails at load time without it. (Same install in CI: `.github/workflows/portability.yml` "Linux native runtime libraries", and `README.md` "Linux Debian/Ubuntu อาจต้องติดตั้ง `python3.10-venv libgomp1 libglu1-mesa`".)
- **`libglu1-mesa`** — `libGLU.so.1`. The OpenSees binary links the OpenGL Utility library (legacy graphics/visualization subsystems are compiled into it even though this app only uses file recorders). Missing `libGLU.so.1` is the classic OpenSeesPy-on-Linux import error.
- Nothing else: no X11 server (Agg), no external OpenSees executable (bundled in the wheel), no compilers (all wheels).

### 5.3 Thread/BLAS pinning (deliberate, everywhere)

`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1` is set in `run.py`, re-set inside every worker (`app/worker.py` top), re-forced in the child env at dispatch (`app/engine.launch`), set in `START_FRAGILITY.bat`, in CI, and as Dockerfile ENV. Parallelism comes **only** from running N independent worker processes (default `min(7, cpu_count)`, cap `cpu_count`; persisted in `state/settings.json`). Keep this in any dockerized variant — oversubscribed BLAS threads would corrupt the "7 native slots" mental model and slow the solver.

## 6. Runtime layout — what gets written where (Docker-critical)

`ROOT = Path(app/common.py).resolve().parents[1]` — **everything is ROOT-relative; there are no hardcoded absolute host paths in runtime code.** Windows drive letters (`D:\QCE-NAS\…`) appear only in committed *provenance records* (`assets/motions.json` `raw_source_path`, `validation/dashboard_http.json` snapshot) — historical evidence, never touched at runtime.

### 6.1 `state/` (per-campaign durable state — compose bind-mounts `./state:/app/state`)

```
state/
├─ server.lock            OS file lock (fcntl flock / msvcrt) — single controller
├─ dashboard.json         {url, pid, production_autostart:false}
├─ settings.json          chosen worker count (persisted across restarts)
├─ current.json           → active run dir name
└─ run-YYYYmmdd-HHMMSS-<hex6>/        (created by Engine.__init__)
   ├─ run.json            frozen plan + run_hash + environment + code_hashes
   ├─ events.jsonl        append-only audit log
   ├─ stop.flag           graceful-stop marker (worker pauses at point boundary)
   ├─ queue.sqlite        jobs + model_reports tables, WAL  → also -wal / -shm
   ├─ jobs/<job-id>/
   │   ├─ job.json        dispatched payload (input to worker)
   │   ├─ receipt.json / error.json     worker outcome (hash-pinned)
   │   ├─ result.json (modal) | curve.json (curve)
   │   ├─ native-<ns>.log raw OpenSees worker stdout+stderr
   │   ├─ activity.json   1 Hz heartbeat during NLTHA (drives dashboard progress)
   │   ├─ native_scratch/ worker-private cwd: OpenSees recorder .out files (deleted after commit)
   │   └─ point-NNN/      one per committed IM level:
   │       <AXIS>.npz (floor histories), <AXIS>-members.npz (member envelopes),
   │       <AXIS>.json (per-point native record), component.json (hash-pinned summary),
   │       legacy_original_result.json (floor-5 imports only)
   ├─ reports/<stories>-<building_id>/  per-model HTML/PNG/CSV/JSON/manifest
   ├─ queue_snapshot.sqlite, job_ledger.json        (export)
   ├─ fragility_summary.csv, campaign_summary.json, research_inventory.json,
   │  index.html, height-{5,6,7,8}-fragility.png    (campaign export)
```

Implications: the SQLite WAL trio lives inside the volume; crash-safety and resume depend on it. The 10 GiB disk-reserve guard watches `shutil.disk_usage(state)` (`app/engine.py`, `minimum_free_gib=10.`) — with the compose bind mount that is the host filesystem.

### 6.2 `runtime_storage/` (caches + temp — compose bind-mounts `./runtime_storage:/app/runtime_storage`)

`app/common.py` sets `FRAGILITY_POC_RUNTIME_ROOT=$ROOT/runtime_storage` and `FRAGILITY_ALLOW_SYSTEM_DRIVE=1` **before** anything imports `fragility_poc`; importing the package runs `src/fragility_poc/runtime_storage.configure_runtime_storage()`, which creates and redirects: `tmp/` (`TEMP`/`TMP`/`TMPDIR` and `tempfile.tempdir` — this is where the batched-dynamic `TemporaryDirectory` recorder files go), `cache/matplotlib` (`MPLCONFIGDIR`), `cache/joblib`, `cache/xdg`, `cache/pip`, `cache/node`, `cache/numba`, `cache/torch`, `jupyter/` (`src/fragility_poc/runtime_storage.py` lines 52–86). Without the env override the default would still resolve to `ROOT/runtime_storage`; the override makes the container path explicit. Additionally, each `curve` worker `chdir`s into `jobs/<id>/native_scratch/` so OpenSees' native relative writes never land in `/app` (`app/worker.py` lines 91–96), and `native_paths.py` forces recorder paths to ASCII-relative form.

### 6.3 Writes inside the source tree (not volumes)

- `validation/package_verification.json` (`app/verify.py`), `validation/results_audit.json` (`app/audit_results.py`), `validation/source_routing_and_relocation.json` (`tests/portability_no_analysis.py`) — overwrite committed evidence files. Inside a container these land in the image layer and evaporate with the container; harmless, but if you want the artifacts, add a volume or copy them out.
- `.downloads/` — `tools/prepare_data.py` cache (gitignored **and** dockerignored, so it is never in the image).
- `assets/legacy_seeds/` — ~643 MB of floor-5 seed data written by `tools/prepare_data.py`. Gitignored but **not** dockerignored: if present on the host at build time it is baked into the image via `COPY . ./`; if absent, it must be mounted or downloaded at runtime (the app refuses nothing at startup, but floor-5 curve jobs will fail at seed import without it — `bundle_manifest.json:missing_research_data_instruction` says "Run `python tools/prepare_data.py` before Start").

### 6.4 Platform assumptions baked into code

- POSIX/Windows branches are both present and switch on `os.name`/`sys.platform`: file lock (`fcntl` vs `msvcrt`), process identity (`psutil` vs kernel32 `ctypes`), `subprocess.CREATE_NO_WINDOW` via `getattr(..., 0)` default. Linux needs nothing special.
- Worker cwd is always `ROOT`; all job/asset paths are passed as absolute paths resolved through `common.resolve()`, which rejects anything escaping `ROOT` ("Path outside portable application").
- `safe_host()` requires the HTTP `Host` header to be exactly `127.0.0.1:<port>` for **GET as well as POST** (`app/server.py`) — a loopback-only design; the `--bind 0.0.0.0` flag exists solely "for a container mapped to host loopback" (argparse help text).

## 7. Docker / Ubuntu implications (summary)

**Single container, multi-process — not multi-service.** One Python process runs the HTTP controller + queue engine (`run.py` → `app/server.py` → `Engine`); up to N `app/worker.py` subprocesses run OpenSees. There is no broker, no database server (SQLite in `state/`), no shared volume between services. Do not split it: the controller and workers communicate only through the filesystem under the run dir, and worker adoption depends on seeing the same PIDs/filesystem.

- **Port**: single HTTP port, default **8788** (`--port`). `compose.yml` publishes `127.0.0.1:8788:8788` (loopback-only, correct for this app — the dashboard has auth-by-token but is designed for SSH-tunnel access, `README.md`). Container CMD binds `0.0.0.0` (`Dockerfile` CMD: `python -B -X utf8 run.py --no-browser --bind 0.0.0.0`).
- **Access quirk to document for users**: the server 403s any request whose `Host` header isn't exactly `127.0.0.1:8788`. Browsing `http://localhost:8788` will be rejected; the URL must be `http://127.0.0.1:8788`.
- **Volumes (already in `compose.yml`, keep them)**: `./state:/app/state` (all durable campaign data, SQLite WAL, worker receipts/points/reports) and `./runtime_storage:/app/runtime_storage` (TEMP/MPLCONFIGDIR/etc.). Both are in `.dockerignore` so they never enter the build context. Losing `state/` loses the campaign; there is no other persistence.
- **Graceful stop**: `stop_grace_period: 30s` is already set. There is no SIGTERM handler; the safe path is the UI's Stop (writes `stop.flag`, workers pause at the next point boundary, receipts commit) — README instructs Stop → wait `active=0` before `docker compose down`. Ungraceful kill is survivable by design (point-level commit + receipt validation + live-worker adoption), but produces `needs_review` entries.
- **Image contents**: `COPY . ./` brings `app/`, `src/`, `src_variants/`, `assets/` (incl. `legacy_seeds/` if present at build time), tools, tests, and `bundle_manifest.json`. **Do not mount host source over `/app`** — the SHA-256 preflight at Start (`Engine.start` → `common.verify_assets`, all 2,616 files) fails on any divergence ("Portable package changed"), and `run.code_hashes` additionally freezes `app/`+`src/`+`src_variants/` per run. Build from a pristine checkout (`git clone` + `tools/prepare_data.py` before `docker build` if you want seeds baked in).
- **Platform pinned to `linux/amd64`**: `Dockerfile` `FROM --platform=linux/amd64 python:3.10-slim-bookworm` and `compose.yml` `platform: linux/amd64`, because `openseespylinux==3.5.1.3` ships an x86_64 binary only. On ARM (Apple Silicon/ARM servers) it runs under emulation — README warns this can be slower and is explicitly the supported path ("ไม่รองรับทุก OS/architecture โดยอัตโนมัติ"). On Ubuntu x86_64 this is native and ideal.
- **System packages**: only `libgomp1` + `libglu1-mesa` (see §5.2). Python 3.10 exactly (`tools/doctor.py` asserts it; the image supplies it).
- **External network access**: none at runtime except `tools/prepare_data.py` (one-time download from `github.com` release asset, or `--archive` offline). The dashboard, workers, and reports need no egress.
- **Resources**: CPU parallelism = worker count (1–`cpu_count` of the container, default `min(7, cpu_count)`, persisted in `state/settings.json`); each worker is single-threaded BLAS by design. RAM: one OpenSees+numpy interpreter per worker plus recorder-file streaming (member envelopes are streamed from disk, not held in memory — `app/worker.py` "Stream raw member recorder rows"). Disk: points retain full floor-history NPZs at **every** committed IM level for 2,000 models × 6–8 curves — tens of GB scale; the controller auto-stops at a 10 GiB free-space reserve and asks for a bigger disk (message in `app/engine.loop`). Put `./state` on roomy host storage.
- **Cross-origin/platform guards that are inert on Linux**: `msvcrt`/`ctypes`-kernel32 branches (Windows-only), `Path.drive == 'C:'` check in `runtime_storage.py` (Windows-only; `FRAGILITY_ALLOW_SYSTEM_DRIVE=1` is set anyway by `app/common.py`), `CREATE_NO_WINDOW` (defaults to 0 elsewhere). `.gitattributes` (`* -text`, LF for `.sh`/`Dockerfile`/`.yml`) exists precisely so checkout OS cannot change any hashed byte — relevant only if you build from a fresh clone on Windows.
- **Locale**: `python -X utf8` everywhere; results are written with `ensure_ascii=True` JSON; the repo path may contain Thai (`ψ/` is this docs folder), and the worker's ASCII-scratch enforcement keeps native OpenSees I/O safe regardless (`app/worker.py:ascii_recorder`).

### Quick reference — dockerized runbook as-is

```bash
python3 tools/prepare_data.py          # host: fetch+verify 643 MB floor-5 seeds (once; bakes into image or mount assets/legacy_seeds)
docker compose up --build              # builds python:3.10-slim-bookworm + libgomp1 + libglu1-mesa
# open exactly http://127.0.0.1:8788   (Host header must be 127.0.0.1:8788)
# press Start (preflight SHA-256 of 2,616 files runs first), set workers, monitor state/
```

Sources: `Dockerfile`, `compose.yml`, `.dockerignore`, `requirements.txt`, `run.py`, `app/{server,engine,worker,common,policy,components,legacy_seeds,reports,verify,audit_results}.py`, `app/dashboard.html`, `assets/{catalog,motions}.json`, `assets/grouped_model_loader.py`, `assets/legacy_seed_index.json`, `assets/models/*storey/{manifest,build_status}.json`, `src/fragility_poc/{__init__,runtime_storage,native_paths,structural,ground_motion,fishbone_ida,fishbone_v4_grouped_frame,fishbone_v5_batched_dynamic}.py`, `src_variants/5storey/fragility_poc/__init__.py`, `tools/{prepare_data,doctor}.py`, `tests/*`, `.github/workflows/portability.yml`, `{bundle_manifest,component_method,source_variants}.json`, `distribution/{crossplatform_declaration,legacy_seed_data}.json`, `validation/*.json`, `{README,README_TH}.md`, `.gitattributes`, `.gitignore`.
