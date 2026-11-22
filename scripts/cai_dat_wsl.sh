#!/usr/bin/env bash
# Cài môi trường huấn luyện trong WSL2 Ubuntu (chạy MỘT lần, bên trong Ubuntu):
#   cd "/mnt/d/hien/Dai hoc/Nam 3/RL/Mario-and-Peaches" && bash scripts/cai_dat_wsl.sh
# venv đặt ở ~/venv-mario (ổ Linux, nhanh hơn /mnt/d và không đụng .venv của Windows).
set -euo pipefail

sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-venv git tmux

python3 -m venv ~/venv-mario
source ~/venv-mario/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q -e .   # biên dịch nes-py, mất vài phút

python -c "import numpy, torch, nes_py; print('numpy', numpy.__version__, '| torch', torch.__version__, '| nes_py ok')"
echo "Xong. Mỗi lần mở Ubuntu: source ~/venv-mario/bin/activate"
