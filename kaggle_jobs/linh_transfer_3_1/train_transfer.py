"""Transfer the selected level 1-1 policy to level 3-1 for a 1M-step scout run."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


COMMIT = "f16a43bb3c1891e78bd150c6cbe0e4916e125aa0"
LEVEL = "3-1"
REPO = Path("/tmp/mario-ppo-linh-transfer-3-1")
RUN = Path("/kaggle/working/linh-transfer-3-1-1m")
RESUME = REPO / "best_models/1-1/xuat-phat-1-1.pt"


print("cpu_count:", os.cpu_count(), flush=True)
print("python:", sys.version.split()[0], flush=True)
print("commit:", COMMIT, flush=True)
subprocess.run(
    ["git", "clone", "--quiet", "https://github.com/studie-e/Mario-and-Peaches.git", str(REPO)],
    check=True,
)
subprocess.run(["git", "-C", str(REPO), "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO), "numpy<2"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO, check=True)

command = [
    sys.executable,
    "-m",
    "mario_ppo.train",
    f"level={LEVEL}",
    "device=cpu",
    "seed=2",
    "algo.gamma=0.99",
    "algo.num_envs=16",
    "torch_threads=2",
    "algo.num_steps=256",
    "algo.total_steps=1000000",
    "checkpoint_every=500000",
    "eval_episodes=5",
    "video=true",
    f"resume={RESUME}",
    f"run_dir={RUN}",
]
print("command:", command, flush=True)
started = time.perf_counter()
subprocess.run(command, cwd=REPO, check=True)

with (RUN / "progress.csv").open(newline="") as handle:
    last = list(csv.DictReader(handle))[-1]
summary = {
    "commit": COMMIT,
    "level": LEVEL,
    "source_checkpoint": str(RESUME.relative_to(REPO)),
    "seed": 2,
    "gamma": 0.99,
    "num_envs": 16,
    "torch_threads": 2,
    "total_steps": 1000000,
    "global_step": int(last["global_step"]),
    "sps": float(last["sps"]),
    "x_pos": float(last["x_pos"]),
    "flag_rate": float(last["flag_rate"]),
    "wall_seconds": round(time.perf_counter() - started, 1),
    "run_dir": str(RUN),
}
(Path("/kaggle/working") / "linh-transfer-3-1-summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print("LINH_TRANSFER_COMPLETE", json.dumps(summary), flush=True)
