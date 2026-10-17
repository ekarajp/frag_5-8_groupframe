# Ubuntu server deployment (Docker)

Run the Component IDA / Fragility dashboard on a headless Ubuntu x86-64 server and reach it
from a browser on your LAN/VLAN. Everything runs in one container; results stay on the host
via bind mounts.

## Prerequisites

- **x86-64 host** (`uname -m` → `x86_64`). The pinned OpenSeesPy 3.5.1.3 native engine is
  x86_64-only; ARM hosts would run under slow emulation and are not supported.
- **Docker Engine 23+ and Docker Compose v2** (BuildKit is the default there — the build uses
  a `--mount=type=cache` download cache). Check: `docker --version`, `docker compose version`.
- **Outbound HTTPS** on the server for the first build: it downloads the immutable floor-5
  seed archive (~643 MB) from the GitHub release and SHA-256-verifies every file. You do
  **not** need Python on the host — the build does this inside the image. For an offline
  server, build the image elsewhere and `docker save` / `docker load` it.
- **~10 GiB free** on the disk holding `state/` (the engine enforces a 10 GiB reserve) plus
  ~2 GB for the image.

## First deployment

```bash
git clone https://github.com/ekarajp/frag_5-8_groupframe.git
cd frag_5-8_groupframe

# 1. Configure access (see table below)
printf 'FRAGILITY_BIND=0.0.0.0\nFRAGILITY_ALLOWED_HOSTS=192.168.1.6:8788\n' > .env

# 2. Create the bind-mount dirs BEFORE first `up`, owned by UID 1000
#    (docker would auto-create them as root, and the container runs as UID 1000)
mkdir -p state runtime_storage
sudo chown -R 1000:1000 state runtime_storage

# 3. Build (downloads ~643 MB seeds on first run) and start
docker compose up -d --build
docker compose ps        # wait for status (healthy)
```

Then open **`http://192.168.1.6:8788`** from a machine on the same LAN. The URL's host part
must match an entry in `FRAGILITY_ALLOWED_HOSTS` exactly (Host-header check).

## Configuration (`.env` next to `compose.yml`)

| Variable | Default | Meaning |
|---|---|---|
| `FRAGILITY_BIND` | `127.0.0.1` | Host interface the port is published on. Use `0.0.0.0` (all interfaces) or a specific IP. **Empty falls back to loopback.** |
| `FRAGILITY_ALLOWED_HOSTS` | *(empty)* | Comma-separated exact `Host:port` values the dashboard accepts, e.g. `192.168.1.6:8788` or `192.168.1.6:8788,100.92.162.38:8788`. `127.0.0.1:<port>` is always allowed. IPv6 literals use bracket form: `[2001:db8::5]:8788`. |
| `FRAGILITY_CPUS` | `10` | Hard CPU quota for the container (all worker processes + controller share it). Set below the host core count to leave room for other work; recreate with `docker compose up -d` after changing. |

The same list can be set in-process with `run.py --allowed-host HOST:PORT` (repeatable,
comma-splittable) for non-Docker use.

## Resource limits & log rotation

- **CPU**: the container is capped by `FRAGILITY_CPUS` (default 10) via the Docker CFS
  quota — analyses cannot starve the host's other services no matter how many workers run.
  Each worker is single-threaded (BLAS/OpenMP pinned to 1) and the controller ≈ 1 core, so
  set the dashboard **Workers ≈ quota − 1** when Starting (more workers than quota just
  time-share it, finishing nothing faster).
- **Memory is not capped** — size the worker count against what the host can spare and
  watch `docker stats` during the first hours of a campaign.
- **Container logs** (`docker compose logs`) use the json-file driver rotated at
  **3 × 10 MB** — bounded disk use with recent history preserved.
- Files under `state/run-*/` (`events.jsonl`, per-job `native-*.log`, points, reports) are
  **campaign audit data, not logs** — they are intentionally not rotated or deleted; the
  engine's 10 GiB disk reserve covers them.

## Daily operation

```bash
docker compose ps              # health, port mapping
docker compose logs -f         # controller log (startup lines list allowed Host values)
docker compose restart         # restart controller (state persists)
docker compose down            # stop (see graceful stop below)
docker compose up -d           # start again
```

**Graceful stop matters.** Before `docker compose down` (or a host reboot) during an active
campaign: press **Stop safely** in the dashboard and wait until **active=0** (workers finish
the current component point, bounded by a 2 h/direction guard, usually minutes). Then `down`.

- `docker stop` / `compose down` sends SIGTERM → the controller now shuts down cleanly
  (writes `stop.flag`, commits SQLite). Workers still running get SIGKILL after the 30 s
  grace period. That is **survivable by design** — on the next Start, jobs resume from
  hash-verified points; interrupted ones are flagged `needs_review` for the audit tool
  instead of being silently trusted.
- Never run two controllers against the same `state/` (enforced by a file lock).

## Campaign freshness rule

`bundle_manifest.json` SHA-256-freezes the shipped source + assets, and each campaign's
`run.json` freezes the manifest hash it started with. **Any change to the image content
(including this repo's own code updates) makes an old campaign refuse Start with
"Bundle manifest changed since this campaign was frozen".** That is integrity working as
intended: archive the old `state/`, start a fresh campaign with the new image, and keep the
old source checkout if you need bit-identical continuation.

To update the app: `git pull && docker compose up -d --build` — expect the rule above for
any in-flight campaign.

## Updating / moving the server

Finish or pause the campaign (Stop → active=0 → `down`), then copy **the whole `state/`
folder** together with the same source/assets version, and resume on the new machine after
checking hashes (the dashboard/worker re-verify automatically). Cross-platform
floating-point differences are recorded, not assumed identical.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Browser gets **403** JSON | Host header mismatch — browse exactly `http://<allowed-host>:8788`; `localhost:8788` fails unless `localhost:8788` is in `FRAGILITY_ALLOWED_HOSTS` |
| Container exits / permission denied writing `state/` | Bind dirs owned by root: `sudo chown -R 1000:1000 state runtime_storage` |
| Port already allocated | Another service on 8788 — change the host side: `FRAGILITY_BIND` pins the interface; edit `compose.yml` ports for a different host port (and add the matching `host:port` to allowed hosts) |
| Health `starting` for long | `docker compose logs -f`; first boot runs seed preflight only at Start, readiness itself is seconds |
| Build fails at `RUN --mount=type=cache` | Legacy builder without BuildKit — upgrade Docker, or `DOCKER_BUILDKIT=1`, or replace that line with `RUN python tools/prepare_data.py && rm -rf .downloads` |
| Want plain `docker run` | `docker run -d --init --platform linux/amd64 -p 8788:8788 -e FRAGILITY_ALLOWED_HOSTS=<ip>:8788 -v $PWD/state:/app/state -v $PWD/runtime_storage:/app/runtime_storage <image>` |

## Security posture (read this once)

Exposing the dashboard to a LAN is a deliberate choice. The token and Origin checks remain,
**but anyone who can reach the port with the right Host header can read `GET /api/status`,
which returns the control token — and therefore has full control of the campaign.** Treat
the VLAN as trusted; keep `FRAGILITY_BIND` on loopback (the default) if that assumption ever
feels wrong, and use an SSH tunnel instead (`ssh -L 8788:127.0.0.1:8788 server`).
