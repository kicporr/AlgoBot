# ============================================================
# bocik — Production Docker image (multi-stage, Linux x86_64)
# ============================================================

FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for TA-Lib build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib C library — single-thread to avoid race condition on ARM/SBC
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make -j1 \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# ── Runtime Stage ────────────────────────────────────────────

FROM python:3.12-slim

LABEL name="bocik"
LABEL description="Multi-asset algorithmic trading bot"

WORKDIR /app

# Copy TA-Lib from builder
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

# Python deps (minimal runtime set)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt \
    && rm -rf /root/.cache/pip /tmp/* \
    && find /usr/local/lib -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib -name "*.pyc" -delete 2>/dev/null || true \
    && find /usr/local/lib -name "*.pyo" -delete 2>/dev/null || true

# App code (runtime-only modules)
COPY config/ config/
COPY models/ models/
COPY data/ data/
COPY dashboard/ dashboard/
COPY features/ features/
COPY strategies/ strategies/
COPY ensemble/ ensemble/
COPY risk/ risk/
COPY execution/ execution/
COPY monitoring/ monitoring/
COPY scripts/watchdog.py scripts/
COPY orchestrator.py .
COPY __init__.py .

# Default: paper trading (safe)
CMD ["python", "orchestrator.py", "--mode", "paper"]

# Health check: lightweight endpoint, no DB queries
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -sf http://localhost:8088/api/health || exit 1
