#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "  Vast MuZero Install"
echo "=========================================="

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: run this script as root on the Vast instance"
    exit 1
fi

PROJECT_SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT_SRC

echo ""
echo "[1/9] Base system packages"
export DEBIAN_FRONTEND=noninteractive
export PIP_BREAK_SYSTEM_PACKAGES=1
apt-get update
apt-get install -y \
    software-properties-common \
    curl \
    ca-certificates \
    git \
    htop \
    ninja-build \
    tmux \
    xvfb \
    ffmpeg \
    libgl1 \
    libglx-mesa0 \
    libglu1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    x11-utils \
    pciutils \
    procps \
    unzip \
    rsync

echo ""
echo "[2/9] Python 3.10"
if ! command -v python3.10 >/dev/null 2>&1; then
    add-apt-repository ppa:deadsnakes/ppa -y
    apt-get update
fi

PY_PACKAGES=(python3.10 python3.10-venv python3.10-dev)
if apt-cache show python3.10-distutils >/dev/null 2>&1; then
    PY_PACKAGES+=(python3.10-distutils)
fi
apt-get install -y "${PY_PACKAGES[@]}"

if ! python3.10 -m pip --version >/dev/null 2>&1; then
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 - --ignore-installed
fi

python3.10 -m pip install --no-cache-dir --upgrade --ignore-installed pip setuptools wheel packaging

cat > /usr/local/bin/python <<'EOF'
#!/usr/bin/env bash
exec python3.10 "$@"
EOF
chmod +x /usr/local/bin/python

cat > /usr/local/bin/pip <<'EOF'
#!/usr/bin/env bash
exec python3.10 -m pip "$@"
EOF
chmod +x /usr/local/bin/pip

echo "python3.10: $(python3.10 --version)"
echo "python:     $(python --version)"

echo ""
echo "[3/9] PyTorch CUDA wheels"
python -m pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "[4/9] Project Python dependencies"
python -m pip install --no-cache-dir \
    gymnasium \
    stable-retro \
    torchrl \
    einops \
    wandb \
    tqdm \
    moviepy \
    imageio \
    imageio-ffmpeg

python -m pip install --no-cache-dir --upgrade --ignore-installed --only-binary cryptography \
    cryptography \
    pyOpenSSL \
    certifi
python -m pip check

echo ""
echo "[5/9] Xvfb"
pkill -f "Xvfb :99" 2>/dev/null || true
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 &
sleep 2
export DISPLAY=:99
xdpyinfo -display :99 >/dev/null

echo ""
echo "[6/9] Environment"
mkdir -p /workspace/wandb /workspace/.torchinductor /workspace/evals /workspace/model_checkpoints

cat > /workspace/vast_env.sh <<'EOF'
#!/usr/bin/env bash
export DISPLAY=:99
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor
export WANDB_DIR=/workspace/wandb

if [ -d /workspace/marioRL_3.10/src ]; then
    export PROJECT_SRC=/workspace/marioRL_3.10/src
elif [ -d /workspace/src ]; then
    export PROJECT_SRC=/workspace/src
else
    PROJECT_FILE=$(find /workspace -maxdepth 5 -type f -name muzero_train.py 2>/dev/null | head -1)
    if [ -n "$PROJECT_FILE" ]; then
        export PROJECT_SRC=$(dirname "$PROJECT_FILE")
    fi
fi

if [ -n "${PROJECT_SRC:-}" ]; then
    export PYTHONPATH="$PROJECT_SRC:${PYTHONPATH:-}"
fi

ulimit -n 1048576 2>/dev/null || ulimit -n 65535 2>/dev/null || true
EOF
chmod +x /workspace/vast_env.sh

if ! grep -q "/workspace/vast_env.sh" ~/.bashrc; then
    cat >> ~/.bashrc <<'EOF'

# === Vast MuZero Environment ===
source /workspace/vast_env.sh
if ! pgrep -x Xvfb >/dev/null 2>&1; then
    Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 &
fi
# === End Vast MuZero Environment ===
EOF
fi

echo ""
echo "[7/9] Helper scripts"
cat > /workspace/vast_restart.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if ! pgrep -x Xvfb >/dev/null 2>&1; then
    Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 &
    sleep 2
fi

source /workspace/vast_env.sh
exec /workspace/vast_verify.sh "$@"
EOF
chmod +x /workspace/vast_restart.sh

cat > /workspace/install_rom.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [ -f /workspace/vast_env.sh ]; then
    source /workspace/vast_env.sh
fi

export DISPLAY="${DISPLAY:-:99}"
if ! pgrep -x Xvfb >/dev/null 2>&1; then
    Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 &
    sleep 2
fi

if [ -n "${1:-}" ]; then
    ROM_DIR="$1"
elif [ -d /workspace/SuperMarioWorld-Snes ]; then
    ROM_DIR=/workspace/SuperMarioWorld-Snes
elif [ -d /workspace/SuperMarioWorld-SNES ]; then
    ROM_DIR=/workspace/SuperMarioWorld-SNES
else
    ROM_DIR=$(find /workspace -maxdepth 4 -type d \( -name "SuperMarioWorld-Snes" -o -name "SuperMarioWorld-SNES" \) 2>/dev/null | head -1)
fi

if [ -z "${ROM_DIR:-}" ] || [ ! -d "$ROM_DIR" ]; then
    echo "ERROR: SuperMarioWorld-Snes folder not found"
    echo "Usage: /workspace/install_rom.sh /path/to/SuperMarioWorld-Snes"
    exit 1
fi

for required in rom.sfc data.json; do
    if [ ! -f "$ROM_DIR/$required" ]; then
        echo "ERROR: missing $ROM_DIR/$required"
        exit 1
    fi
done

RETRO_DATA=$(python -c "import retro; print(retro.data.path())")
TARGET_DIR="${RETRO_DATA}/stable/SuperMarioWorld-Snes"
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"
rsync -a "$ROM_DIR"/ "$TARGET_DIR"/

python - <<'PY'
import retro

for state in ("YoshiIsland2", "DonutPlains1", "VanillaDome4", "Bridges2"):
    env = retro.make("SuperMarioWorld-Snes", state=state, render_mode="rgb_array")
    env.reset()
    env.close()
    print(f"{state}: OK")
PY
echo "ROM installed to $TARGET_DIR"
EOF
chmod +x /workspace/install_rom.sh

cat > /workspace/run_muzero_smoke.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

source /workspace/vast_env.sh
cd "${PROJECT_SRC:-/workspace/marioRL_3.10/src}"

python muzero_train.py \
    --total_steps 500 \
    --workers 2 \
    --batch_size 8 \
    --mcts_sims 2 \
    --mcts_depth 2 \
    --mcts_backend tensor \
    --search_batch_size 8 \
    --max_episode_steps 200 \
    --min_replay_transitions 100 \
    --levels YoshiIsland2 \
    --no_wandb
EOF
chmod +x /workspace/run_muzero_smoke.sh

cat > /workspace/run_muzero_train.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

source /workspace/vast_env.sh
cd "${PROJECT_SRC:-/workspace/marioRL_3.10/src}"

python muzero_train.py --preset first_real "$@"
EOF
chmod +x /workspace/run_muzero_train.sh

cat > /workspace/vast_verify.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

source /workspace/vast_env.sh
exec bash "$PROJECT_SRC/vast_verify.sh" "$@"
EOF
chmod +x /workspace/vast_verify.sh

echo ""
echo "[8/9] ROM auto-install"
ROM_DIR=$(find /workspace -maxdepth 4 -type d \( -name "SuperMarioWorld-Snes" -o -name "SuperMarioWorld-SNES" \) 2>/dev/null | head -1 || true)
if [ -n "$ROM_DIR" ] && [ -f "$ROM_DIR/rom.sfc" ]; then
    /workspace/install_rom.sh "$ROM_DIR"
else
    echo "ROM folder not found yet. Upload it, then run:"
    echo "  /workspace/install_rom.sh /workspace/SuperMarioWorld-SNES"
fi

echo ""
echo "[9/9] Verification"
/workspace/vast_verify.sh

echo ""
echo "=========================================="
echo "  Install Complete"
echo "=========================================="
echo "Next:"
echo "  source /workspace/vast_env.sh"
echo "  /workspace/install_rom.sh /workspace/SuperMarioWorld-SNES"
echo "  /workspace/vast_verify.sh --require-rom"
echo "  /workspace/run_muzero_smoke.sh"
echo "  /workspace/run_muzero_train.sh"
