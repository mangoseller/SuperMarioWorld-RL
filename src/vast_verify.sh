#!/usr/bin/env bash
set -euo pipefail

REQUIRE_ROM=0
RUN_SMOKE=0

for arg in "$@"; do
    case "$arg" in
        --require-rom)
            REQUIRE_ROM=1
            ;;
        --smoke)
            RUN_SMOKE=1
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--require-rom] [--smoke]"
            exit 2
            ;;
    esac
done

if [ -f /workspace/vast_env.sh ]; then
    # shellcheck disable=SC1091
    source /workspace/vast_env.sh
fi

if [ -z "${PROJECT_SRC:-}" ]; then
    PROJECT_FILE=$(find /workspace "$(pwd)" -maxdepth 5 -type f -name muzero_train.py 2>/dev/null | head -1 || true)
    if [ -n "$PROJECT_FILE" ]; then
        PROJECT_SRC=$(dirname "$PROJECT_FILE")
        export PROJECT_SRC
    fi
fi

if [ -z "${PROJECT_SRC:-}" ] || [ ! -f "$PROJECT_SRC/muzero_train.py" ]; then
    echo "ERROR: could not find project source containing muzero_train.py"
    exit 1
fi

export PYTHONPATH="$PROJECT_SRC:${PYTHONPATH:-}"
cd "$PROJECT_SRC"

echo "=========================================="
echo "  Vast MuZero Verification"
echo "=========================================="
echo "Project: $PROJECT_SRC"

echo ""
echo "[1/7] Python commands"
command -v python3.10
command -v python
python3.10 --version
python --version

PY_MINOR=$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
if [ "$PY_MINOR" != "3.10" ]; then
    echo "ERROR: python resolves to $PY_MINOR, expected 3.10"
    exit 1
fi

echo ""
echo "[2/7] Python packages"
python -m pip --version
python -m pip check
python - <<'PY'
import importlib

mods = [
    "torch",
    "torchvision",
    "gymnasium",
    "retro",
    "torchrl",
    "einops",
    "wandb",
    "tqdm",
    "moviepy",
    "imageio",
]
for name in mods:
    mod = importlib.import_module(name)
    version = getattr(mod, "__version__", "ok")
    print(f"{name}: {version}")
PY

echo ""
echo "[3/7] CUDA"
python - <<'PY'
import torch as t

print("torch:", t.__version__)
print("cuda_available:", t.cuda.is_available())
print("cuda_version:", t.version.cuda)
print("device_count:", t.cuda.device_count())
for i in range(t.cuda.device_count()):
    props = t.cuda.get_device_properties(i)
    print(f"gpu_{i}: {props.name}, {props.total_memory / 1024**3:.1f} GiB")
if not t.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available to PyTorch")
PY

echo ""
echo "[4/7] Display"
export DISPLAY="${DISPLAY:-:99}"
if ! pgrep -x Xvfb >/dev/null 2>&1; then
    Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 &
    sleep 2
fi
if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    echo "Xvfb OK on $DISPLAY"
else
    echo "ERROR: Xvfb is not usable on $DISPLAY"
    exit 1
fi

echo ""
echo "[5/7] MuZero imports and syntax"
python -m py_compile \
    muzero_train.py \
    environment.py \
    wrappers.py \
    rewards.py \
    curriculum.py \
    muzero/config.py \
    muzero/network.py \
    muzero/mcts.py \
    muzero/self_play.py \
    muzero/reanalyse.py \
    muzero/replay_buffer.py \
    muzero/eval.py
python - <<'PY'
from muzero.config import MuZeroConfig, first_real_run_config
from muzero.network import MuZeroNetwork

config = MuZeroConfig()
network = MuZeroNetwork(config)
params = sum(p.numel() for p in network.parameters())
run_config = first_real_run_config()
print("MuZeroNetwork params:", f"{params:,}")
print("first_real workers:", run_config.self_play_workers)
print("first_real backend:", run_config.mcts_backend)
PY

echo ""
echo "[6/7] ROM"
ROM_OK=0
if python - <<'PY'
import retro

env = retro.make("SuperMarioWorld-Snes", state="YoshiIsland2", render_mode="rgb_array")
env.reset()
env.close()
PY
then
    ROM_OK=1
    echo "ROM OK: SuperMarioWorld-Snes/YoshiIsland2 loads"
else
    echo "ROM missing or not installed. Run:"
    echo "  /workspace/install_rom.sh /workspace/SuperMarioWorld-Snes"
fi

if [ "$REQUIRE_ROM" -eq 1 ] && [ "$ROM_OK" -ne 1 ]; then
    exit 1
fi

echo ""
echo "[7/7] Optional smoke"
if [ "$RUN_SMOKE" -eq 1 ]; then
    if [ "$ROM_OK" -ne 1 ]; then
        echo "ERROR: --smoke requires a working ROM"
        exit 1
    fi
    /workspace/run_muzero_smoke.sh
else
    echo "Skipped. Use --smoke after ROM installation."
fi

echo ""
echo "Verification complete."
