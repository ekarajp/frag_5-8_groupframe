# CODE SNIPPETS — frag_5-8_groupframe (dockerization prep)

- **Date:** 2026-10-17 17:17
- **Repo:** `/home/phaith/ghq/github.com/ekarajp/frag_5-8_groupframe`
- **Purpose:** Key machinery excerpts for fully dockerizing this Python 3.10 / OpenSeesPy seismic fragility app on Ubuntu.
- **Shape of the app:** `run.py` → `app/server.py` (stdlib `ThreadingHTTPServer`, port 8788) → `app/engine.py` (SQLite job queue) → one `app/worker.py` subprocess per native OpenSees job. The `src/fragility_poc/` package (~50k lines) holds the scientific engine; 2000 grouped-frame models are **frozen data files**, not generated at runtime.

---

## 1. Main entry point

### 1.1 `run.py` (whole file, 10 lines) — thread pinning + backend before anything else

`run.py:1-10`

```python
"""Cross-platform entry point. Never starts analyses automatically."""
import os,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLBACKEND','Agg')
sys.path.insert(0,str(ROOT/'app'))
if __name__=='__main__':
    runpy.run_path(str(ROOT/'app/server.py'),run_name='__main__')
```

**Why it matters:** Every BLAS/OpenMP library is pinned to 1 thread *per process* before numpy/matplotlib import. Parallelism comes only from running N worker processes. `ROOT` is the resolved parent of `run.py` → in Docker (`WORKDIR /app`, `COPY . ./`) that is `/app`; all path guards are relative to it. `server.py` is exec'd via `runpy`, so CLI args go through `sys.argv` unchanged.

### 1.2 `START_FRAGILITY.sh` — POSIX launcher with venv preference

`START_FRAGILITY.sh:1-7`

```sh
#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ -x .venv/bin/python ]; then
    exec .venv/bin/python -B -X utf8 run.py "$@"
fi
exec python3 -B -X utf8 run.py "$@"
```

### 1.3 `START_FRAGILITY.bat` — Windows equivalent (redundant thread vars here)

`START_FRAGILITY.bat:3-11`

```bat
cd /d "%~dp0"
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -B -X utf8 "%~dp0run.py"
) else (
  python -B -X utf8 "%~dp0run.py"
)
```

**Why it matters:** `-B` (no `.pyc`) and `-X utf8` (UTF-8 mode) are used on **every** launch path, including the Dockerfile CMD and worker subprocesses. UTF-8 mode is a hard assumption (repo is delivered inside Thai-named folders).

### 1.4 `app/server.py` `main()` — flags, bind, single-instance lock, browser

`app/server.py:19-49`

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8788)
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--state')
    parser.add_argument('--bind',choices=('127.0.0.1','0.0.0.0'),default='127.0.0.1',
                        help='0.0.0.0 only for a container mapped to host loopback')
    args = parser.parse_args()
    state = Path(args.state or ROOT/'state').resolve()
    state.mkdir(parents=True,exist_ok=True)
    # OS lock released on crash, unlike a stale heartbeat lock file.
    lock = (state/'server.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        if not args.no_browser:
            webbrowser.open(f'http://127.0.0.1:{args.port}')
        print('This portable campaign already has a dashboard/controller.')
        return
    engine = Engine(state)
    atexit.register(engine.close)
    token = secrets.token_urlsafe(32)
```

`app/server.py:132-147`

```python
    httpd = ThreadingHTTPServer((args.bind,args.port),Handler)
    write(state/'dashboard.json',{'url':f'http://127.0.0.1:{args.port}','pid':os.getpid(),
                                'production_autostart':False})
    if not args.no_browser:
        webbrowser.open(f'http://127.0.0.1:{args.port}')
    print(f'Dashboard ready: http://127.0.0.1:{args.port} — waiting for Start',flush=True)
    try:
        httpd.serve_forever(poll_interval=.3)
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()
        httpd.server_close()
```

**Why it matters (Docker):**
- `--bind` only accepts the two loopback/all-interfaces choices; the Dockerfile CMD passes `--no-browser --bind 0.0.0.0`.
- Single-instance enforcement is an **OS advisory lock** (`fcntl.flock` on Linux) on the *shared state volume* — two containers mounting the same `state/` cannot both run.
- **No SIGTERM handler exists.** Only `KeyboardInterrupt` (SIGINT) is caught; `atexit`/`finally` cleanup runs on SIGINT or normal exception, not on Docker's default SIGTERM stop.

### 1.5 Existing container files (already shipped)

`Dockerfile:1-8`

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

`compose.yml:1-10`

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

`requirements.txt:3-11` — fully pinned, OS-conditional solver wheels:

```
numpy==1.26.4
scipy==1.15.3
matplotlib==3.10.9
pandas==2.3.3
joblib==1.5.3
psutil==7.0.0
openseespy==3.5.1.3
openseespywin==3.5.1.3; sys_platform == "win32"
openseespylinux==3.5.1.3; sys_platform == "linux"
```

**Why it matters:** `libgomp1` + `libglu1-mesa` are the two native deps OpenSeesPy needs on slim-bookworm. Version pinning is *asserted* at runtime (see §4.4 `tools/doctor.py`).

---

## 2. The HTTP server core (`app/server.py`, stdlib only)

### 2.1 Routing — GET

`app/server.py:77-101`

```python
        def do_GET(self):
            if not self.safe_host():
                return self.send(403,{'error':'Local dashboard Host required'})
            path = urlparse(self.path).path
            try:
                if path == '/api/status':
                    return self.send(200,{**engine.snapshot(),'token':token,'export':export_status})
                if path == '/api/reports':
                    return self.send(200,[{'name':p.parent.name,'path':'/results/'+p.relative_to(engine.run_dir).as_posix()}
                        for p in sorted((engine.run_dir/'reports').glob('*/index.html'))])
                if path.startswith('/results/'):
                    p = (engine.run_dir / unquote(path[len('/results/'):])).resolve()
                    if not p.is_relative_to(engine.run_dir) or not p.is_file():
                        return self.send(404,{'error':'File not found'})
                    return self.send(200,p.read_bytes(),mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
                if path.startswith('/reference/'):
                    p=(ROOT/'assets'/unquote(path[len('/reference/'):])).resolve()
                    if not p.is_relative_to(ROOT/'assets') or not p.is_file():
                        return self.send(404,{'error':'Reference not found'})
                    ...
                if path == '/':
                    return self.send(200,(ROOT/'app/dashboard.html').read_bytes(),'text/html; charset=utf-8')
                self.send(404,{'error':'Not found'})
            except Exception as e:
                self.send(500,{'error':str(e)})
```

### 2.2 Routing — POST (how analyses are triggered)

`app/server.py:102-131`

```python
        def do_POST(self):
            if not self.safe_host() or self.headers.get('X-Runner-Token') != token:
                return self.send(403,{'error':'Invalid local request token'})
            origin = self.headers.get('Origin')
            if origin and origin != f'http://127.0.0.1:{args.port}':
                return self.send(403,{'error':'Cross-origin request refused'})
            try:
                length = int(self.headers.get('Content-Length','0'))
                if length > 4096:
                    raise ValueError('Request too large')
                body = json.loads(self.rfile.read(length) or b'{}')
                if self.path == '/api/start':
                    engine.start(body.get('workers',7))
                elif self.path == '/api/stop':
                    engine.stop()
                elif self.path == '/api/workers':
                    engine.set_workers(body['workers'])
                elif self.path == '/api/export':
                    if engine.processes or engine.active:
                        raise ValueError('Stop and drain before creating a consistent full research inventory')
                    if export_status['state'] != 'working':
                        engine.final_export_state='queued'
                        threading.Thread(target=export,daemon=True).start()
                else:
                    return self.send(404,{'error':'Unknown action'})
                self.send(200,{'ok':True})
            except (ValueError,KeyError) as e:
                self.send(400,{'error':str(e)})
            except Exception as e:
                self.send(500,{'error':str(e)})
```

### 2.3 The Host-header pin (critical Docker fact)

`app/server.py:75-76`

```python
        def safe_host(self):
            return self.headers.get('Host') == f'127.0.0.1:{args.port}'
```

**Why it matters:** Every request (GET and POST) is rejected with 403 unless the HTTP `Host` header is *exactly* `127.0.0.1:8788`. Behind the compose mapping `127.0.0.1:8788:8788` this works because the browser sends that exact host. But: no other hostname, no container IP, no reverse proxy with a rewritten Host, no `-H "Host: localhost"` will work. Tests lock this in — `tests/test_dashboard_portable.py:24-26` asserts an unauthenticated `/api/start` gets `403`.

### 2.4 Response hardening

`app/server.py:63-74`

```python
        def send(self,code,data,ctype='application/json; charset=utf-8'):
            if isinstance(data,(dict,list)):
                data = json.dumps(data,ensure_ascii=True,allow_nan=False).encode()
            ...
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
```

`Handler.log_message` is overridden to `pass` (`app/server.py:61-62`) — the server is silent; logging goes to `events.jsonl` via the engine.

---

## 3. The OpenSees analysis engine (IDA runner)

### 3.1 Process topology

Dashboard process (`run.py` → `server.py` → `Engine`) **never imports OpenSees** (`app/common.py:1`: *"No OpenSees import in the dashboard process"*). Each job = one `app/worker.py` subprocess:

`app/engine.py:157-182` (dispatch)

```python
    def launch(self, row):
        job = json.loads(row['payload'])
        dest = self.run_dir / 'jobs' / row['id']
        dest.mkdir(parents=True,exist_ok=True)
        ...
        write(dest / 'job.json', job)
        env = dict(os.environ)
        env.update({key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        log = (dest / ('native-' + str(time.time_ns()) + '.log')).open('wb')
        try:
            p = subprocess.Popen([sys.executable, '-B', '-X','utf8',str(self.worker_script),
                '--job',str(dest / 'job.json'),'--run',str(self.run_dir / 'run.json')],
                cwd=ROOT, env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        finally:
            log.close()
        ident = identity(p.pid)
        self.processes[row['id']] = {'process':p,'identity':ident}
```

**Why it matters:** workers are `sys.executable` children with `cwd=ROOT`; `CREATE_NO_WINDOW` degrades to `0` on Linux. Worker stdout/stderr land in `jobs/<id>/native-*.log`.

### 3.2 The 2000 grouped-frame model variants — frozen data, replayed command lists

The models are **not generated at runtime**. 2000 gzip-JSON construction recipes ship in `assets/models/{5,6,7,8}storey/models/B-*.json.gz`, indexed by `assets/catalog.json` (verified: 2000 entries; each row = `building_id`, `model_hash`, `path`, `sha256`, `bytes`, `stories`, `recipe`, `candidate_hash`; 500 per height). A command whitelist loader replays them into a fresh OpenSees domain:

`assets/grouped_model_loader.py:7-32, 34-44`

```python
CONSTRUCTION_COMMANDS = frozenset({
    'wipe', 'model', 'node', 'fix', 'mass', 'equalDOF', 'geomTransf',
    'uniaxialMaterial', 'section', 'patch', 'layer', 'fiber',
    'beamIntegration', 'element', 'timeSeries', 'pattern', 'load', 'eleLoad',
})

def read_model(path, expected_sha256=None):
    path = Path(path)
    if expected_sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Model file checksum mismatch')
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        model = json.load(handle)
    if model.get('schema') != 'portable-grouped-construction-v1':
        raise ValueError('Unsupported model schema')
    ...
    for name, args in commands:
        if name not in CONSTRUCTION_COMMANDS or not isinstance(args, list):
            raise ValueError('Forbidden or malformed construction command: ' + name)
    return model

def build_model(path, direction, ops, expected_sha256=None):
    """Populate one fresh OpenSees domain; gravity/damping/analysis are caller work."""
    model = read_model(path, expected_sha256)
    unit = model['directions'][direction.upper()]
    for name, args in unit['commands']:
        getattr(ops, name)(*args)
    return unit['model_info'], unit['audit']
```

**Why it matters:** Building a model = replaying stored OpenSees CLI commands (`getattr(ops, name)(*args)`); nothing is calibrated or regenerated in the container. Integrity is enforced by `bundle_manifest.json` SHA-256 verification before any analysis (§5, §7).

### 3.3 Worker main — job kinds, receipts, hard exits

`app/worker.py:286-319`

```python
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    p.add_argument('--run', required=True)
    args = p.parse_args()
    jobpath = Path(args.job).resolve()
    out = jobpath.parent
    try:
        job, run = read(jobpath), read(args.run)
        if digest({k:v for k,v in run.items() if k != 'run_hash'}) != run['run_hash']:
            raise ValueError('Run declaration hash mismatch')
        for row in run.get('code_hashes',[]):
            if sha(resolve(row['path'])) != row['sha256']:
                raise ValueError('Frozen scientific/application source changed: '+row['path'])
        from common import configure_scientific_source
        configure_scientific_source(job['model']['stories'])
        write(out / 'environment.json', environment())
        result = modal(job, run, out) if job['kind'] == 'modal' else curve(job, run, out)
        # A receipt is written only after all original outputs and validation commit.
        if job['kind'] == 'curve' and (out/'curve.json').exists():
            result['curve_sha256'] = sha(out/'curve.json')
        write(out / 'receipt.json', {**result, 'job_id': job['id'], 'run_hash': run['run_hash'],
            'finished': time.time(), 'pid': os.getpid()})
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except BaseException:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        write(out / 'error.json', {'error': error, 'time': time.time(), 'pid': os.getpid()})
        os._exit(1)
```

**Why it matters:** `os._exit()` bypasses interpreter shutdown so OpenSees' C++ layer cannot hang the container; every failure produces `error.json` + exit 1, every success a hash-anchored `receipt.json`. Workers also re-hash all source files listed in the frozen run plan before doing science.

### 3.4 The IDA curve loop — hunt/bisection, resumable points

`app/worker.py:166-199` (decision loop inside `curve()`)

```python
    from legacy_seeds import import_seeds
    import_seeds(job,run,out,recipe)
    points = []
    # Only complete, hash-verified pair points can be resumed; no partial-prefix reuse.
    for p in sorted(out.glob('point-*/component.json')):
        r = read(p)
        for name, checksum in r['file_hashes'].items():
            if sha(p.parent / name) != checksum:
                raise ValueError('Saved point hash mismatch')
        ...
    try:
        while True:
            decision = decide(points, run['policy'])
            write(out / 'curve.json', {...'points': points, **decision})
            if decision['state'] != 'continue':
                return decision
            if (out.parents[1] / 'stop.flag').exists():
                return {'state': 'paused'}
            im = decision['next_im_g']
            sf = im / job['unscaled_sa_g']
            ...
            pointdir = out / f'point-{len(points):03d}'
            ...
            result = dyn._run_direction_batched(artifact, model['building'],
                {'model': recipe['model_config'], 'building': recipe['building_config']},
                axis, accelerations[axis][1], record_dt_s=float(motion['dt_s']),
                scale_factor=sf, damping_ratio=.05, damping_model='modal',
                dynamic_instability_drift=.04,
                maximum_runtime_s=run['policy']['runtime_s_per_direction'],
                minimum_substep_factor=64, chunk_steps=10)
```

`app/policy.py:6-34` — the bisection brain:

```python
def decide(points, policy):
    caps = estimate_limit_state_capacities(points, THRESHOLDS,
                          bracket_tolerance=policy['bracket_tolerance'])
    if points and points[-1]['status'] == 'numerical_failure':
        return {'state': 'needs_review', 'reason': 'numerical_failure_not_a_crossing', 'capacities': caps}
    if all(c['certified'] for c in caps.values()):
        return {'state': 'complete', 'capacities': caps}
    if len(points) >= policy['maximum_points_per_curve']:
        return {'state': 'needs_review', 'reason': 'point_guard_no_automatic_extension', 'capacities': caps}
    ...
    return {'state': 'continue', 'next_im_g': im, 'capacities': caps}
```

Default policy (`app/engine.py:16-24`): `initial_im_g .05`, IM range `.001..20`, hunt multiplier 1.5, bracket tolerance .05, **`maximum_points_per_curve: 100`**, **`runtime_s_per_direction: 7200.0`** (2 h guard per direction — see §7 shutdown discussion), thresholds IO/LS/CP = .01/.02/.04 (MIDR).

### 3.5 The NLTHA core — `_run_direction_batched`

`src/fragility_poc/fishbone_v5_batched_dynamic.py:100-207` (abridged)

```python
def _run_direction_batched(artifact, building, config, direction, acceleration_g, *,
    record_dt_s, scale_factor, damping_ratio, damping_model,
    dynamic_instability_drift, maximum_runtime_s, minimum_substep_factor, chunk_steps):
    ...
    info, audit = build_grouped_frame_model(artifact, building, config, normalized, ...)
    gravity_error = run_grouped_frame_gravity(info, audit)
    damping = _apply_damping(artifact, info, normalized, damping_ratio, damping_model)
    acceleration = np.asarray(acceleration_g, dtype=float) * G_STD * scale_factor
    ops = _ops()
    ...
    ops.timeSeries("Path", series_tag, "-dt", record_dt_s, "-values", *acceleration)
    ops.pattern("UniformExcitation", pattern_tag, dof, "-accel", series_tag)
    _configure_transient()
    ...
    recovery_plan = (
        ("Newton", "EnergyIncr", 1.0e-7, 40),
        ("NewtonLineSearch", "EnergyIncr", 1.0e-7, 60),
        ("KrylovNewton", "EnergyIncr", 1.0e-7, 80),
        ("ModifiedNewton", "EnergyIncr", 1.0e-6, 120),
        ("KrylovNewton", "NormDispIncr", 1.0e-6, 120),
    )
    ...
    while float(ops.getTime()) < duration - tolerance_time:
        if time.perf_counter() - started > maximum_runtime_s:
            numerical_failure = True
            collapse_basis.append("runtime_guard")
            break
        ...
        code = int(ops.analyze(steps, step_dt))
        if code == 0:
            continue
        ...  # substep factors 1,2,4,...,64 x recovery_plan fallbacks, single-step retries
```

The OpenSees import itself is wrapped, `src/fragility_poc/structural.py:1292-1302`:

```python
def _ops() -> Any:
    try:
        import openseespy.opensees as ops
    except ImportError as exc:
        raise RuntimeError(
            "OpenSeesPy is not installed. Install the locked project environment "
            "with: python -m pip install -e .[test]") from exc
    from .native_paths import install_native_paths
    install_native_paths(ops)
    return ops
```

Gravity + modal sanity (`src/fragility_poc/fishbone_v4_grouped_frame.py:1242-1257`): `ops.analyze(10)` static gravity with `LoadControl 0.1`, base-reaction equilibrium error must be < 1% or the worker raises.

### 3.6 Parallelism — process fan-out, not joblib

- **No joblib / multiprocessing in the dashboard path.** Concurrency = up to N worker subprocesses; default **7**, hard cap `os.cpu_count()`:
  `app/engine.py:39-42`

  ```python
  self.workers = min(7, os.cpu_count() or 1)
  self.worker_limit = max(1, os.cpu_count() or 1)
  if (self.state/'settings.json').is_file():
      self.workers=max(1,min(self.worker_limit,int(read(self.state/'settings.json')['workers'])))
  ```

  In Docker, `os.cpu_count()` sees the container's visible CPUs.
- Each worker process is single-threaded with all BLAS vars forced to `1` twice (run.py at import, worker.py:3-4, and again via engine env at spawn).
- The only thread pools in `app/`: `ThreadPoolExecutor(max_workers=1)` for report/export I/O (`engine.py:43`) and `ThreadPoolExecutor(max_workers=8)` purely for SHA-256 asset verification I/O (`common.py:124`).
- Controller poll loop: `app/engine.py:245-296`, refill cadence `time.sleep(.15)` (`engine.py:296`); picks pending jobs `ORDER BY CASE kind WHEN 'curve' THEN 0 ELSE 1 END,rowid` (`engine.py:268`).
- `joblib` appears only in legacy scientific modules (`src/fragility_poc/active_ida.py:17,121`, `ml.py`) that the portable campaign does not invoke.
- Stale-crash recovery adopts *live* orphans instead of respawning — `app/engine.py:97-104`:

  ```python
  # A reopened app never duplicates a live orphan native worker.
  for row in self.db.execute("SELECT * FROM jobs WHERE status='running'").fetchall():
      ident = json.loads(row['pid_identity'] or 'null')
      if ident and identity(ident['pid']) == ident:
          self.processes[row['id']] = {'process':None, 'identity':ident}
          self.event('adopted_live_worker',job=row['id'],identity=ident)
      else:
          self.finish(row, None)
  ```

  Identity = PID + process create-time + exe, via psutil on POSIX (`app/common.py:55-67`).

---

## 4. Configuration & environment

### 4.1 CLI flags (complete list)

| Command | Flags | Where |
|---|---|---|
| `run.py` → server | `--port` (default 8788), `--no-browser`, `--state PATH`, `--bind {127.0.0.1,0.0.0.0}` (default loopback) | `app/server.py:20-26` |
| `app/worker.py` | `--job PATH --run PATH` | `app/worker.py:287-290` |
| `app/audit_results.py` | `--run DIR`, `--deep` | `app/audit_results.py:9` |
| `tools/doctor.py` | `--solver-import` | `tools/doctor.py:8` |
| `tools/prepare_data.py` | `--archive PATH` (else downloads) | `tools/prepare_data.py:36` |

### 4.2 Environment variables the code reads/sets

| Var | Where | Effect |
|---|---|---|
| `OMP_NUM_THREADS` `OPENBLAS_NUM_THREADS` `MKL_NUM_THREADS` `NUMEXPR_NUM_THREADS` = 1 | `run.py:5-6`, `app/worker.py:3-4`, `app/engine.py:169`, Dockerfile ENV, CI workflow env | pin native math threads per process |
| `MPLBACKEND=Agg` | `run.py:7` (setdefault), Dockerfile, `.github/workflows/portability.yml`, re-forced via `matplotlib.use('Agg')` in `app/reports.py:31,96` | headless plotting |
| `FRAGILITY_POC_RUNTIME_ROOT` | set `app/common.py:12`; read `src/fragility_poc/runtime_storage.py:35` | redirects all library caches/temp into `<ROOT>/runtime_storage` |
| `FRAGILITY_ALLOW_SYSTEM_DRIVE=1` | set `app/common.py:13`, `src/fragility_poc/config.py:32-33`; read `runtime_storage.py:43` | bypasses the "no Drive C" workstation guard (dead code on Linux, but proves portability intent) |
| `COMPONENT_SCIENTIFIC_VARIANT` | set `app/common.py:159`; read `app/common.py:111` | provenance tag: `5storey-original` / `6-8-original` / `dashboard-no-native` |
| `FRAGILITY_POC_COMPONENT_RESPONSES` | read `src/fragility_poc/config.py:34-52` | override enabling/disabling member-response recording (validated against `{1,0,true,false,enabled,disabled,on,off}`) |
| `FRAGILITY_POC_NODE_EXE`, `FRAGILITY_POC_NODE_MODULES` | read `src/fragility_poc/reporting.py:1365-1379` | optional Node runtime for workbook artifact tool (legacy path; falls back to a hardcoded `~/.cache/codex-runtimes/...` that will **not exist** in a slim container) |
| Set *by* `runtime_storage.py:67-86`: `TEMP`, `TMP`, `TMPDIR`, `MPLCONFIGDIR`, `JOBLIB_TEMP_FOLDER`, `XDG_CACHE_HOME`, `PIP_CACHE_DIR`, `npm_config_cache`, `NUMBA_CACHE_DIR`, `TORCH_HOME`, `JUPYTER_RUNTIME_DIR`, plus `tempfile.tempdir` | `src/fragility_poc/runtime_storage.py:52-86` | every program-owned write under `runtime_storage/` |

### 4.3 Config / data files read at runtime

- `bundle_manifest.json` (519 KB) — SHA-256 + byte size of every shipped file; hashed whole by `Engine.start` and per-file by `verify_assets` (`app/common.py:113-135`, `app/engine.py:121-122`).
- `assets/catalog.json` — the 2000-model plan (`app/engine.py:60,91`).
- `assets/motions.json` — 23 CMS ground-motion pairs with per-component paths + SHA-256s (`app/worker.py:68,83`, `app/components.py`).
- `assets/legacy_seed_index.json` — floor-5 reuse seeds (`app/legacy_seeds.py:10`).
- `component_method.json` — campaign declaration (schema `portable-component-5-8-v3`; `total_models: 2000`, `automatic_start: false`) — documentation-level, read by humans/verify tooling.
- `source_variants.json` — documents the height routing (`"5" → src_variants/5storey`, `"6"/"7"/"8" → src`) and the 5 scientific files that differ between the variants.
- `state/settings.json` — persisted worker count (`app/engine.py:41-42,154`).
- `state/current.json` — pointer to the active `run-<ts>-<id>` dir (`app/engine.py:48-55`).
- `distribution/legacy_seed_data.json` — download manifest for `tools/prepare_data.py`.

### 4.4 OS-conditional code (all of it)

- `app/server.py:36-41` — `msvcrt.locking` vs `fcntl.flock` for the single-instance lock.
- `app/common.py:57-94` — process identity: `psutil` on non-Windows vs raw `ctypes`/kernel32 on Windows.
- `src/fragility_poc/runtime_storage.py:43` — `storage.drive.upper() == "C:"` guard (Windows-only concept; no-op on Linux).
- `src/fragility_poc/reporting.py:1405-1419` — `cmd /c mklink /J` junction vs `symlink_to` (legacy workbook tool only).
- `tools/doctor.py:9-16` — hard assertions used by CI:

  ```python
  assert sys.version_info[:2]==(3,10),'Use Python 3.10 x64, or Docker linux/amd64.'
  assert platform.system() in ('Windows','Linux'),'On macOS use Docker linux/amd64 with the frozen solver version.'
  ...
  native='openseespywin' if sys.platform=='win32' else 'openseespylinux'
  assert importlib.metadata.version(native)==expected   # '3.5.1.3'
  ```
- `requirements.txt:10-11` — platform markers select `openseespywin` / `openseespylinux`.
- `subprocess` `creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)` (`app/engine.py:175`) — safe no-op on Linux.

### 4.5 Source variant routing (5-storey vs 6-8)

`app/common.py:138-160`

```python
def configure_scientific_source(stories):
    """One isolated worker selects its original height-specific source snapshot."""
    if stories not in (5,6,7,8):raise ValueError('Only heights 5 through 8 belong to this package')
    if any(n=='fragility_poc' or n.startswith('fragility_poc.') for n in sys.modules):
        raise ValueError('Select the scientific snapshot before importing a scientific module')
    source=ROOT/'src_variants/5storey' if stories==5 else ROOT/'src'
    sys.path.insert(0,str(source))
    if stories==5:
        # ... adapter: route the old workstation-only I/O helper through the portable one
        ...
    os.environ['COMPONENT_SCIENTIFIC_VARIANT']='5storey-original' if stories==5 else '6-8-original'
    return source
```

**Why it matters:** both `src/` and `src_variants/5storey/` (a full copy of `fragility_poc`) must ship in the image; `sys.path` order per worker decides which scientific snapshot runs.

---

## 5. Filesystem behavior (every runtime read/write)

### 5.1 Atomic-write pattern (used everywhere state is committed)

`app/common.py:32-47`

```python
def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=True, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(3):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            ...
```

(same idea in `src/fragility_poc/io_utils.py:30-44`; npz saves via `save_history` `app/worker.py:17-25`).

### 5.2 `state/` (compose volume `./state:/app/state`)

| Path | Writer | Purpose |
|---|---|---|
| `state/server.lock` | `server.py:30` | `flock` single-instance lock |
| `state/dashboard.json` | `server.py:133` | url + pid + `production_autostart: false` |
| `state/settings.json` | `engine.py:154` | worker count |
| `state/current.json` | `engine.py:75` | pointer `{'run': run_dir.name}` |
| `state/run-<ts>-<hex>/run.json` | `engine.py:74` | frozen plan: policy + SHA-256s of bundle/catalog/motions + `code_hashes` + `run_hash` |
| `.../queue.sqlite` (+`-wal`,`-shm`) | `engine.py:79-88` | jobs + model_reports tables, `PRAGMA journal_mode=WAL` |
| `.../events.jsonl` | `engine.py:108-110` | append-only audit log |
| `.../stop.flag` | `engine.py:145,294` | graceful-stop signal read by controller and workers |
| `.../jobs/<job_id>/job.json`, `receipt.json`, `error.json`, `environment.json`, `native-*.log`, `activity.json` | worker/engine | per-job artifacts; `activity.json` is the live heartbeat (`worker.py:153-161`, ≥1/s during `analyze`) |
| `.../jobs/<job_id>/point-NNN/` | worker | committed IDA points: `component.json` (with `file_hashes`), `<axis>.npz` histories, `<axis>.json`, `<axis>-members.npz` envelopes; legacy seeds add `legacy_original_result.json` |
| `.../jobs/<job_id>/native_scratch/` | worker | private cwd for OpenSees recorders (`worker.py:91-96`, `os.chdir` + restore in `finally`) |
| `.../reports/<model_id>/` | `app/reports.py:33-91` | `index.html`, `results.json`, `ida_points.csv` (utf-8-sig), `component_capacities.csv` (utf-8-sig), **`ida_fragility.png`** (matplotlib Agg, dpi 150), `manifest.json` |
| `.../fragility_summary.csv`, `campaign_summary.json`, `height-<n>-fragility.png`, `index.html`, `research_inventory.json`, `queue_snapshot.sqlite`, `job_ledger.json` | `app/reports.py:94-123`, `engine.py:230-244` | final export |

Disk-space guard, `app/engine.py:263-266`:

```python
                        free = shutil.disk_usage(self.state).free/2**30
                        if free < self.minimum_free_gib:      # 10 GiB default, engine.py:27
                            self.stop()
                            self.reason = f'Disk reserve reached ({free:.1f} GiB); move the whole stopped package to a larger disk, then resume'
```

### 5.3 `runtime_storage/` (compose volume `./runtime_storage:/app/runtime_storage`)

Forced by `app/common.py:12` (`FRAGILITY_POC_RUNTIME_ROOT`), consumed by `src/fragility_poc/runtime_storage.py:52-86`: creates and points caches at `runtime_storage/tmp`, `cache/{matplotlib,joblib,xdg,pip,node,numba,torch}`, `jupyter/`. `common.py:13` (`FRAGILITY_ALLOW_SYSTEM_DRIVE=1`) is what lets this run outside the original Windows D: layout.

### 5.4 Other writes

- `validation/package_verification.json` — `app/verify.py:44` (read-only package audit; asserts 2000 models / 4000 construction replays / 23 pairs / **0 analyses / no solver import**).
- `validation/results_audit.json` — `app/audit_results.py:40`.
- `.downloads/` cache + `assets/legacy_seeds/**` — `tools/prepare_data.py:42-47` (**only network touch in the repo**; downloads a zip if seed files are absent/invalid).
- matplotlib PNG output only (Agg); no display ever opened.

---

## 6. Error handling examples

### 6.1 Native worker failure → `error.json` + nonzero exit (§3.3, `worker.py:312-316`)

`except BaseException` → full traceback to both stderr (captured into `native-*.log`) and `error.json`, then `os._exit(1)`.

### 6.2 Controller translates exit/receipt into job states

`app/engine.py:184-215`

```python
    def finish(self, row, code):
        dest = self.run_dir / 'jobs' / row['id']
        state, reason = 'needs_review', 'Native process ended without a valid committed receipt'
        try:
            receipt = read(dest / 'receipt.json')
            if receipt['job_id'] != row['id'] or receipt['run_hash'] != self.run['run_hash']:
                raise ValueError('Receipt identity mismatch')
            if code not in (0,None):
                raise ValueError(f'Observed nonzero process exit {code}; receipt requires incident audit')
            state = receipt['state']
            ...
        except Exception as error:
            state, reason = 'needs_review', str(error)
        self.db.execute('UPDATE jobs SET status=?,reason=?,finished=? WHERE id=?',
                        ('pending' if state == 'paused' else state,reason,time.time(),row['id']))
        self.event('job_finished',job=row['id'],state=state,reason=reason,observed_exit=code)
```

Failures surface on the dashboard via `/api/status` → `'failures': [rows with status=='needs_review'][:100]` (`engine.py:335`).

### 6.3 Controller crash containment

`app/engine.py:290-295`

```python
            except Exception as e:
                with self.lock:
                    self.active = False
                    self.reason = 'Controller needs review: ' + str(e)
                    write(self.run_dir / 'stop.flag', {'error':str(e)})
                    self.event('controller_error',error=str(e))
```

### 6.4 Report and export errors are recorded, never raised to HTTP

- Per-model reports: `app/engine.py:217-228` catches everything into `model_reports(state='needs_review', error=...)` and `report_errors`.
- Final export: `app/engine.py:242-244` → `final_export_state = 'needs_review: '+str(e)` + `research_inventory_error` event.

### 6.5 HTTP layer mapping

403 host/token/origin (`server.py:79,103-107`), 404 unknown path/file (`server.py:90,95,99`), 400 for `ValueError/KeyError` from engine calls (`server.py:128-129`), 500 catch-all (`server.py:130-131`, `100-101`), 413-style guard via `ValueError('Request too large')` at 4096 bytes (`server.py:110-111`). JSON is serialized with `allow_nan=False`, so NaN can never reach a client.

### 6.6 Numerical failures are data, not exceptions

`app/policy.py:9-10`: a point ending in `numerical_failure` makes the *curve* `needs_review` ("numerical_failure_not_a_crossing") — the runner never auto-retries (`app/engine.py:16-24` policy: *"retain, mark needs_review, no unchanged retry"*; refusal also at `engine.py:163-165`: *"Terminal receipt exists for pending job; automatic retry refused"*).

---

## 7. Docker-relevant code facts

1. **Exact port:** `8788`, hardcoded default (`app/server.py:21`), overridable `--port`. Compose publishes `127.0.0.1:8788:8788`.
2. **Bind address:** `--bind` is restricted to `('127.0.0.1','0.0.0.0')` (`server.py:24-25`); the Dockerfile CMD passes `--bind 0.0.0.0` because inside the container the loopback-only default would be unreachable through the port mapping.
3. **Host-header pin:** `safe_host()` requires `Host == 127.0.0.1:<port>` on *every* request (`server.py:75-79`), and POST additionally demands `Origin == http://127.0.0.1:<port>` + a per-boot `X-Runner-Token` (`server.py:49,103-107`). Consequence: the app is only reachable as `http://127.0.0.1:8788` from the host running the mapping; plain port-forward without that Host header (e.g. remote access, proxies) gets 403. README (line 28) documents the intended remote pattern: SSH tunnel `ssh -L 8788:127.0.0.1:8788`.
4. **Browser auto-open:** `webbrowser.open(...)` fires after successful bind when `--no-browser` is absent (`server.py:135-136`) *and* in the lock-contention early return (`server.py:43-44`). In a container there is no browser; the shipped Dockerfile correctly passes `--no-browser`. Any new entrypoint must keep that flag or tolerate `webbrowser` failing silently.
5. **Single-instance lock is OS-level on the shared volume:** `fcntl.flock` on `state/server.lock` (`server.py:36-41`). Safe across container restarts (kernel releases on crash); a second container on the same `state/` volume prints "already has a dashboard/controller" and exits 0.
6. **Thread limits:** all four BLAS-family env vars forced to `1` in three places (parent, engine-spawned env, worker itself); default worker count `min(7, os.cpu_count())`, max `os.cpu_count()` (`engine.py:39-40`) → size container CPUs accordingly; asset preflight hashing uses 8 I/O threads (`common.py:124`); reports/exports serialize through 1 thread (`engine.py:43`).
7. **Encoding assumptions:** every launcher uses `-B -X utf8`; all JSON/text I/O passes explicit `encoding='utf-8'` (`common.py:19,36`, `io_utils.py:37,48`, `engine.py:109`); CSVs use `utf-8-sig` for Excel (`reports.py:59,63,102`); OpenSees recorder files must have **ASCII relative** paths — the worker chdirs into `native_scratch/` and rewrites/validates recorder paths (`worker.py:91-118`, raises `'Native scratch filename must be ASCII'`). This is how the repo tolerates living under the Thai-named `ψ/` directory. In Docker keep `WORKDIR /app` ASCII anyway.
8. **Signals / long-running requests — why `stop_grace_period: 30s`:**
   - The server handles only `KeyboardInterrupt` (SIGINT); **there is no SIGTERM handler**, so `docker compose stop` SIGTERMs the dashboard and `atexit`/`finally` cleanup does not run.
   - Real graceful stop is application-level: `POST /api/stop` → `engine.stop()` writes `stop.flag` (`engine.py:142-147`) → each worker finishes its *current* NLTHA point and returns `{'state':'paused'}` (`worker.py:190-191`) → controller drains. A single point is bounded only by `runtime_s_per_direction: 7200.0` (2 h) per direction (`engine.py:19`, enforced in `fishbone_v5_batched_dynamic.py:196-199` as a `runtime_guard` numerical failure) — so a "Stop" can legitimately take hours. README (line 40): press Stop, wait `active=0`, then `docker compose down`.
   - `stop_grace_period: 30s` is therefore just a backstop for the dashboard process itself, not the workers: on SIGTERM the container's PID-1 python dies and the container (and its worker children) are torn down with it. Durability is preserved anyway: committed points are hash-verified on resume (`worker.py:167-181`), uncommitted point dirs trigger an audit-required error instead of silent reuse (`worker.py:196-198`), and orphaned live workers are re-adopted by PID identity on restart (`engine.py:97-104`). Workers hard-exit via `os._exit` so no zombie C++ threads linger.
   - `Engine.close()` (`engine.py:343-349`) — `stop()` → controller `join(timeout=3)` → report-pool `shutdown(wait=True)` → DB commit — is what SIGINT triggers.
9. **Startup preflight is heavy:** first `POST /api/start` verifies SHA-256 of every file in `bundle_manifest.json` (~all assets, parallel 8-thread hashing, progress on `/api/status` `preflight`, `engine.py:121-133`, `common.py:113-135`) and refuses if `bundle_manifest.json`'s own hash differs from the frozen run plan. Anything mutating `COPY . ./` content breaks `Start`.
10. **Disks:** hard auto-stop at `< 10 GiB` free on the `state` volume (`engine.py:263-266`); sqlite runs WAL (`engine.py:83`) → the bind-mounted `state/` must be on a filesystem with working `mmap`/fsync (default ext4 fine; avoid some network FSes).
11. **Network at runtime:** none. `tools/prepare_data.py` is the sole downloader and is a build/setup-time step (CI runs it before tests). Models, motions, seeds are frozen files verified by hash.
12. **Runtime pins to assert in the image:** Python `3.10` and `openseespy==3.5.1.3` are hard-asserted (`tools/doctor.py:9-16`); `--platform=linux/amd64` is required (pinned in Dockerfile + compose; openseespylinux has no arm64 wheel). Apt needs `libgomp1`, `libglu1-mesa`.
13. **Healthcheck candidate:** `GET /api/status` returns 200 with JSON (token is included in the payload for the local UI) and needs no auth — only the loopback `Host` header. Idle startup is verified by `tests/test_dashboard_portable.py:22-23`: `status['running'] == False`, `models_total == 2000`, and `/api/start` without token → 403; **opening the dashboard never starts analyses** (`run.py:1`, `server.py:1`, `dashboard.json`'s `production_autostart: False`).
14. **Logs:** the HTTP server is mute (`server.py:61-62`); operational log = `stdout` line `Dashboard ready: ...` (with `PYTHONUNBUFFERED=1` in the image) plus `state/run-*/events.jsonl` and per-job `native-*.log` on the state volume.
