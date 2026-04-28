#!/bin/bash
# vast_setup.sh - Complete setup script for Vast.ai instances
# Run this ONCE when you first create the instance
# After instance restart, run: ./vast_restart.sh

set -e

echo "=========================================="
echo "  Vast.ai Environment Setup Script"
echo "=========================================="

# --- 1. Install Python 3.10 ---
echo ""
echo "[1/8] Installing Python 3.10..."
apt-get update
apt-get install -y software-properties-common curl ca-certificates
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update
apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils

# Install pip for Python 3.10
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 - --ignore-installed

echo "Python installed: $(python3.10 --version)"

# --- 2. Install System Dependencies ---
echo ""
echo "[2/8] Installing system dependencies..."
apt-get install -y \
    git \
    htop \
    ninja-build \
    tmux \
    xvfb \
    ffmpeg \
    libgl1 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    x11-utils \
    pciutils \
    procps \
    unzip

# --- 3. Install Python Packages ---
echo ""
echo "[3/8] Installing Python packages..."
python3.10 -m pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

python3.10 -m pip install --no-cache-dir \
    gymnasium \
    stable-retro \
    torchrl \
    einops \
    wandb \
    tqdm \
    moviepy \
    imageio \
    imageio-ffmpeg

python3.10 -m pip check

# --- 4. Set Up Display (Xvfb) ---
echo ""
echo "[4/8] Starting Xvfb virtual display..."

pkill -f "Xvfb :99" 2>/dev/null || true
sleep 1

Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!
sleep 2

if kill -0 $XVFB_PID 2>/dev/null; then
    echo "✓ Xvfb started (PID: $XVFB_PID)"
else
    echo "✗ WARNING: Xvfb may not have started"
fi

export DISPLAY=:99

# --- 5. Configure Environment ---
echo ""
echo "[5/8] Configuring shell environment..."

cat > /workspace/vast_env.sh << 'EOF'
#!/bin/bash
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
    PROJECT_FILE=$(find /workspace -maxdepth 4 -type f -name muzero_train.py 2>/dev/null | head -1)
    if [ -n "$PROJECT_FILE" ]; then
        export PROJECT_SRC=$(dirname "$PROJECT_FILE")
    fi
fi
if [ -z "$PROJECT_SRC" ] && [ -d /workspace ]; then
    export PROJECT_SRC=/workspace
fi
export PYTHONPATH=${PROJECT_SRC:-/workspace}:$PYTHONPATH
alias python=python3.10
alias pip="python3.10 -m pip"
ulimit -n 1048576 2>/dev/null || ulimit -n 65535 2>/dev/null || true
EOF
chmod +x /workspace/vast_env.sh

cat >> ~/.bashrc << 'EOF'

# === Vast.ai RL Environment ===
source /workspace/vast_env.sh

# Auto-start Xvfb if not running
if ! pgrep -x "Xvfb" > /dev/null; then
    Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
    sleep 1
fi
# === End Vast.ai RL Environment ===
EOF

mkdir -p /workspace/evals /workspace/model_checkpoints
mkdir -p /workspace/wandb /workspace/.torchinductor

# --- 6. Create Restart Script ---
echo ""
echo "[6/8] Creating restart helper script..."

cat > /workspace/vast_restart.sh << 'EOF'
#!/bin/bash
# Run this after instance restart to restore environment

echo "Restoring Vast.ai environment..."

# Start Xvfb if not running
if ! pgrep -x "Xvfb" > /dev/null; then
    echo "Starting Xvfb..."
    Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
    sleep 2
fi

export DISPLAY=:99
source /workspace/vast_env.sh

# Verify Xvfb
if xdpyinfo -display :99 >/dev/null 2>&1; then
    echo "✓ Xvfb running on display :99"
else
    echo "✗ Xvfb may not be running correctly"
fi

# Check ROM installation
echo ""
echo "Checking ROM installation..."
python3.10 -c "
import torch
import retro
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
try:
    env = retro.make('SuperMarioWorld-Snes', state='YoshiIsland2', render_mode='rgb_array')
    env.reset()
    env.close()
    print('✓ ROM installed and working')
except FileNotFoundError:
    print('✗ ROM not found - run install_rom.sh')
"

echo ""
echo "Environment ready! Use 'python3.10' or 'source ~/.bashrc' for aliases."
EOF
chmod +x /workspace/vast_restart.sh

# --- 7. Create ROM Install Script ---
echo ""
echo "[7/8] Creating ROM installation script..."

cat > /workspace/install_rom.sh << 'EOF'
#!/bin/bash
# Install SuperMarioWorld ROM for stable-retro
# Usage: ./install_rom.sh [path_to_SuperMarioWorld-Snes_folder]

set -e

# Find ROM folder
if [ -n "$1" ]; then
    ROM_DIR="$1"
elif [ -d "/workspace/SuperMarioWorld-Snes" ]; then
    ROM_DIR="/workspace/SuperMarioWorld-Snes"
else
    # Search common locations
    ROM_DIR=$(find /workspace -maxdepth 3 -type d -name "SuperMarioWorld-Snes" 2>/dev/null | head -1)
fi

if [ -z "$ROM_DIR" ] || [ ! -d "$ROM_DIR" ]; then
    echo "ERROR: SuperMarioWorld-Snes folder not found!"
    echo ""
    echo "Usage: ./install_rom.sh /path/to/SuperMarioWorld-Snes"
    echo ""
    echo "The folder should contain:"
    echo "  - rom.sfc (REQUIRED)"
    echo "  - data.json (REQUIRED)"
    echo "  - *.state files (YoshiIsland2.state, etc.)"
    exit 1
fi

echo "Found ROM folder: $ROM_DIR"
echo ""

# Check required files
echo "Checking required files..."
if [ ! -f "$ROM_DIR/rom.sfc" ]; then
    echo "✗ ERROR: rom.sfc not found"
    exit 1
fi
echo "✓ rom.sfc"

if [ ! -f "$ROM_DIR/data.json" ]; then
    echo "✗ ERROR: data.json not found"
    exit 1
fi
echo "✓ data.json"

# List state files
echo ""
echo "State files found:"
for state in "$ROM_DIR"/*.state; do
    [ -f "$state" ] && echo "  ✓ $(basename "$state")"
done

# Get retro data path and install
echo ""
echo "Installing ROM..."
RETRO_DATA=$(python3.10 -c "import retro; print(retro.data.path())")
TARGET_DIR="${RETRO_DATA}/stable/SuperMarioWorld-Snes"

# Remove old installation if exists
rm -rf "$TARGET_DIR" 2>/dev/null || true

# Copy everything
mkdir -p "$TARGET_DIR"
cp -r "$ROM_DIR"/* "$TARGET_DIR/"

echo "Installed to: $TARGET_DIR"

# Verify
echo ""
echo "Verifying installation..."

STATES=("YoshiIsland2" "YoshiIsland1" "DonutPlains1")
ALL_OK=true

for state in "${STATES[@]}"; do
    if python3.10 -c "
import retro
env = retro.make('SuperMarioWorld-Snes', state='$state', render_mode='rgb_array')
env.reset()
env.close()
" 2>/dev/null; then
        echo "  ✓ $state"
    else
        echo "  ✗ $state FAILED"
        ALL_OK=false
    fi
done

echo ""
if $ALL_OK; then
    echo "=========================================="
    echo "  ROM Installation Complete!"
    echo "=========================================="
else
    echo "WARNING: Some states failed to load."
    echo "Check that all .state files are present."
fi
EOF
chmod +x /workspace/install_rom.sh

# --- 8. Create MuZero run helpers ---
echo ""
echo "[8/8] Creating MuZero run helper scripts..."

cat > /workspace/run_muzero_smoke.sh << 'EOF'
#!/bin/bash
set -e

source /workspace/vast_env.sh
cd "${PROJECT_SRC:-/workspace/marioRL_3.10/src}"

python3.10 muzero_train.py \
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

cat > /workspace/run_muzero_train.sh << 'EOF'
#!/bin/bash
set -e

source /workspace/vast_env.sh
cd "${PROJECT_SRC:-/workspace/marioRL_3.10/src}"

python3.10 muzero_train.py --preset first_real "$@"
EOF
chmod +x /workspace/run_muzero_train.sh

# --- Try to auto-install ROM if found ---
echo ""
echo "Looking for ROM folder..."

ROM_DIR=$(find /workspace -maxdepth 3 -type d -name "SuperMarioWorld-Snes" 2>/dev/null | head -1)

if [ -n "$ROM_DIR" ] && [ -f "$ROM_DIR/rom.sfc" ]; then
    echo "Found ROM at: $ROM_DIR"
    /workspace/install_rom.sh "$ROM_DIR"
else
    echo "ROM folder not found. After uploading, run:"
    echo "  /workspace/install_rom.sh /path/to/SuperMarioWorld-Snes"
fi

# --- Final Summary ---
echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Python:  $(python3.10 --version)"
echo "PyTorch: $(python3.10 -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python3.10 -c 'import torch; print(torch.cuda.is_available())')"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi
echo "Display: $DISPLAY"
echo ""
echo "Helper scripts created:"
echo "  /workspace/vast_env.sh      - Shared environment exports"
echo "  /workspace/vast_restart.sh  - Run after instance restart"
echo "  /workspace/install_rom.sh   - Install/reinstall ROM"
echo "  /workspace/run_muzero_smoke.sh - Short no-WandB MuZero smoke run"
echo "  /workspace/run_muzero_train.sh - First real MuZero run preset"
echo ""
echo "NEXT STEPS:"
echo "  1. source ~/.bashrc  (to enable python/pip aliases)"
echo "  2. If ROM not installed: ./install_rom.sh /path/to/SuperMarioWorld-Snes"
echo "  3. Set WANDB: export WANDB_API_KEY='your-key'"
echo "  4. Smoke test: /workspace/run_muzero_smoke.sh"
echo "  5. Full run: /workspace/run_muzero_train.sh"
echo ""
echo "After instance RESTART, just run: /workspace/vast_restart.sh"
echo "=========================================="
