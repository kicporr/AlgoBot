# ============================================================
# bocik — Docker image (Linux x86_64)
# ============================================================

FROM python:3.11-slim

LABEL name="bocik"
LABEL description="Multi-asset algorithmic trading bot — 1H/4H"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib (C library) — single-thread build to avoid race condition
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make -j1 \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Default: paper trading (safe)
CMD ["python", "orchestrator.py", "--mode", "paper"]
