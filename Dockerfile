FROM --platform=linux/amd64 python:3.10-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/app/runtime_storage/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglu1-mesa && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY tools/prepare_data.py tools/prepare_data.py
COPY distribution/legacy_seed_data.json distribution/legacy_seed_data.json
RUN --mount=type=cache,target=/app/.downloads python tools/prepare_data.py
COPY . ./
RUN useradd --uid 1000 --create-home fragility
EXPOSE 8788
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD ["python","-c","import urllib.request;urllib.request.urlopen('http://127.0.0.1:8788/',timeout=3)"]
USER fragility
CMD ["python","-B","-X","utf8","run.py","--no-browser","--bind","0.0.0.0"]
