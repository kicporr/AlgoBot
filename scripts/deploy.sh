#!/usr/bin/env bash
# ============================================================
# bocik — Production Deployment Script
# ============================================================
# Usage:
#   ./scripts/deploy.sh              # Fresh deploy (paper mode)
#   ./scripts/deploy.sh --live       # Deploy with live trading
#   ./scripts/deploy.sh --update     # Pull latest, rebuild, restart
#   ./scripts/deploy.sh --status     # Show running containers
#   ./scripts/deploy.sh --logs       # Follow bot logs
#   ./scripts/deploy.sh --down       # Stop everything
# ============================================================

set -euo pipefail

cd "$(dirname "$0")/.."

MODE="paper"
ACTION="deploy"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --live)   MODE="live"; shift ;;
        --update) ACTION="update"; shift ;;
        --status) ACTION="status"; shift ;;
        --logs)   ACTION="logs"; shift ;;
        --down)   ACTION="down"; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Actions ───────────────────────────────────────────────────

case "$ACTION" in

    deploy)
        echo "=== bocik: Fresh Deploy (mode=$MODE) ==="

        # Check for config
        if [ ! -f config/.env ]; then
            echo "ERROR: config/.env not found. Create it from config/.env.example"
            exit 1
        fi

        # Create required directories
        mkdir -p data logs data/cache

        # Pull latest image deps
        docker compose pull 2>/dev/null || true

        # Build and start
        if [ "$MODE" = "live" ]; then
            echo "WARNING: Deploying in LIVE mode. Trades will execute on exchange."
            echo "Press Ctrl+C within 5 seconds to abort..."
            sleep 5
            docker compose run -d --rm bot python orchestrator.py --mode live
        else
            docker compose up -d --build
        fi

        echo ""
        echo "=== bocik deployed ==="
        echo "Dashboard:  http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo 'localhost'):8088"
        echo "Status:     docker compose ps"
        echo "Logs:       docker compose logs -f bot"
        echo "Stop:       docker compose down"
        ;;

    update)
        echo "=== bocik: Update ==="
        git pull --ff-only
        docker compose up -d --build
        docker compose up -d watchdog
        echo "Updated. Check: docker compose logs -f bot"
        ;;

    status)
        docker compose ps
        echo ""
        echo "Health check:"
        curl -s http://localhost:8088/api/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "(bot not reachable)"
        ;;

    logs)
        docker compose logs -f --tail=50 bot
        ;;

    down)
        echo "=== bocik: Shutdown ==="
        docker compose down
        echo "All containers stopped."
        ;;

esac
