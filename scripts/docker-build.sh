#!/usr/bin/env bash
# ============================================================
# bocik — Full Docker Build Script
# 1. Build React dashboard
# 2. Copy to dashboard/ for Python server
# 3. Build Docker image
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1/3 Building React dashboard ==="
cd dashboard-react
npm install --silent 2>/dev/null || true
npm run build
cd ..

echo "=== 2/3 Copying dashboard to Python server ==="
rm -f dashboard/assets/*
cp dashboard-react/dist/index.html dashboard/
cp dashboard-react/dist/assets/* dashboard/assets/

echo "=== 3/3 Building Docker image ==="
docker compose build --no-cache

echo ""
echo "Done. Start with: docker compose up -d"
echo "Dashboard: http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo 'localhost'):8088"
