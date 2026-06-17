#!/bin/bash
# OmniAgent Demo Pro launch script
# Supports port detection and automatic cleanup

set -e

export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL_PATH_ARG=""
if [[ -z "${MODEL_PATH:-}" && "$#" -gt 0 && "$1" != -* ]]; then
    MODEL_PATH_ARG="$1"
    shift
fi

# ============ Default Configuration ============
MODEL_PATH="${MODEL_PATH:-${MODEL_PATH_ARG:-}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.6}"
SHARE="${SHARE:-}"
DEMO_TYPE="${DEMO_TYPE:-pro}"
AUTO_KILL="${AUTO_KILL:-true}"  # Whether to auto-kill processes occupying the port

# ============ Color Definitions ============
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

RAY_TMP_DIR="${RAY_TMP_DIR:-/tmp/ray}"
if [[ -z "${RAY_TMP_DIR}" || "${RAY_TMP_DIR}" == "/" ]]; then
    echo "Invalid RAY_TMP_DIR: ${RAY_TMP_DIR}"
    exit 1
fi
mkdir -p "${RAY_TMP_DIR}"
rm -rf "${RAY_TMP_DIR:?}/"*
echo  "Cleaned Ray temp dir: ${RAY_TMP_DIR}"

# ============ Function Definitions ============

print_banner() {
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                                                            ║"
    echo "║           🎬 OmniAgent Demo Pro                           ║"
    echo "║                                                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_config() {
    echo -e "${BLUE}┌────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│${NC} ${YELLOW}Configuration:${NC}"
    echo -e "${BLUE}│${NC}   Model:      ${GREEN}$MODEL_PATH${NC}"
    echo -e "${BLUE}│${NC}   Address:    ${GREEN}http://$HOST:$PORT${NC}"
    echo -e "${BLUE}│${NC}   GPU Memory: ${GREEN}$GPU_MEMORY_UTIL${NC}"
    echo -e "${BLUE}│${NC}   Demo Type:  ${GREEN}$DEMO_TYPE${NC}"
    if [ -n "$SHARE" ]; then
        echo -e "${BLUE}│${NC}   Public URL: ${GREEN}enabled${NC}"
    fi
    echo -e "${BLUE}└────────────────────────────────────────────────────────────┘${NC}"
    echo ""
}

# Check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is occupied
    else
        return 1  # Port is free
    fi
}

# Get process info occupying the port
get_port_process() {
    local port=$1
    lsof -Pi :$port -sTCP:LISTEN 2>/dev/null | tail -n +2
}

# Kill processes occupying the port
kill_port_process() {
    local port=$1
    local pids=$(lsof -Pi :$port -sTCP:LISTEN -t 2>/dev/null)

    if [ -n "$pids" ]; then
        echo -e "${YELLOW}⚠️  Port $port is occupied by the following process(es):${NC}"
        get_port_process $port
        echo ""

        if [ "$AUTO_KILL" = "true" ]; then
            echo -e "${YELLOW}🔪 Killing process(es): $pids${NC}"
            kill -9 $pids 2>/dev/null || true
            sleep 1

            # Check again
            if check_port $port; then
                echo -e "${RED}❌ Failed to free port $port${NC}"
                exit 1
            else
                echo -e "${GREEN}✅ Port $port is now free${NC}"
            fi
        else
            echo -e "${RED}❌ Port $port is occupied. Set AUTO_KILL=true to auto-kill or free the port manually.${NC}"
            exit 1
        fi
    fi
}

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}🧹 Cleaning up...${NC}"
}

# ============ Main Flow ============

# Switch to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Print banner
print_banner

# Detect and handle port conflicts
echo -e "${BLUE}🔍 Checking port $PORT...${NC}"
if check_port $PORT; then
    kill_port_process $PORT
else
    echo -e "${GREEN}✅ Port $PORT is available${NC}"
fi
echo ""

# Print configuration
print_config

if [ -z "$MODEL_PATH" ]; then
    echo -e "${RED}❌ MODEL_PATH is required. Use: bash demo/launch_demo.sh /path/to/model${NC}"
    exit 1
fi

# Set trap
trap cleanup EXIT

# Select demo file
case "$DEMO_TYPE" in
    pro)
        DEMO_FILE="demo/omniagent_demo_pro.py"
        ;;
    *)
        echo -e "${RED}❌ Unknown DEMO_TYPE: $DEMO_TYPE (use: pro)${NC}"
        exit 1
        ;;
esac

# Check if demo file exists
if [ ! -f "$DEMO_FILE" ]; then
    echo -e "${RED}❌ Demo file not found: $DEMO_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}🚀 Starting $DEMO_TYPE demo...${NC}"
echo -e "${CYAN}   File: $DEMO_FILE${NC}"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Build command
CMD=(
    python "$DEMO_FILE"
    --model_path "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --tensor_parallel_size "$TENSOR_PARALLEL"
    --gpu_memory_utilization "$GPU_MEMORY_UTIL"
)

if [ -n "$SHARE" ]; then
    CMD+=(--share)
fi

if [ "$#" -gt 0 ]; then
    CMD+=("$@")
fi

# Launch
exec "${CMD[@]}"
