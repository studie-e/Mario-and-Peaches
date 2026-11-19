"""Chấm một hoặc nhiều checkpoint bằng ĐÚNG lệnh Mục 4 của đề (20 lượt, --seed 2026).

Mỗi checkpoint được chấm bằng `python -m mario_ppo.evaluate ...` trong một tiến trình riêng, nên kết quả trùng
với cách giảng viên chấm. Kết quả JSON ghi vào danh-gia-tam/, bảng tổng hợp in ra màn hình và nối vào
danh-gia-tam/lich-su.csv để chép sang bảng theo dõi chung.

    python scripts/cham_diem.py --level 1-1 runs/1-1/<run>/checkpoints/step_*.pt
    python scripts/cham_diem.py --level 1-1 --nop <ckpt>     # chấm + ghi video, chép vào nop-bai/ để nộp
"""
import argparse
import csv
import glob
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TAM = Path("danh-gia-tam")
NOP = Path("nop-bai")


def cham(ckpt, level, out, video=None):
    cmd = [sys.executable, "-m", "mario_ppo.evaluate", "--ckpt", str(ckpt), "--level", level,
           "--episodes", "20", "--seed", "2026", "--out", str(out)]
    if video:
        cmd += ["--video", str(video)]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return json.loads(Path(out).read_text(encoding="utf-8"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+", help="đường dẫn hoặc mẫu glob tới các tệp .pt")
    p.add_argument("--level", required=True)
    p.add_argument("--nop", action="store_true",
                   help="ghi video và chép checkpoint, JSON, video, config.yaml/progress.csv/evals.csv vào nop-bai/")
    args = p.parse_args(argv)

    ckpts = sorted({Path(f) for pat in args.ckpts for f in (glob.glob(pat) or [pat])})
    if args.nop and len(ckpts) != 1:
        p.error("--nop chỉ nhận đúng một checkpoint")
    TAM.mkdir(exist_ok=True)

    rows = []
    for ck in ckpts:
        name = f"{args.level}_{ck.parent.parent.name}_{ck.stem}"
        if args.nop:
            (NOP / "danh-gia").mkdir(parents=True, exist_ok=True)
            out, video = NOP / "danh-gia" / f"ket-qua-{args.level}.json", NOP / "danh-gia" / f"{args.level}.mp4"
        else:
            out, video = TAM / f"{name}.json", None
        r = cham(ck, args.level, out, video)
        rows.append({"thoi_diem": f"{datetime.now():%Y-%m-%d %H:%M}", "level": args.level, "ckpt": str(ck),
                     "global_step": r["global_step"], "flag_rate": r["flag_rate"],
                     "mean_x_pos": round(r["mean_x_pos"]), "solved": r["solved"]})
        if args.nop:
            _chep_nop(ck, args.level)

    log = TAM / "lich-su.csv"
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)

    print(f"\n{'checkpoint':<70} {'bước':>10} {'flag':>5} {'x tb':>6}")
    for r in sorted(rows, key=lambda r: (r["flag_rate"], r["mean_x_pos"]), reverse=True):
        print(f"{r['ckpt']:<70} {r['global_step']:>10} {r['flag_rate']:>5.2f} {r['mean_x_pos']:>6}"
              f"{'  VƯỢT' if r['solved'] else ''}")


def _chep_nop(ck, level):
    """Chép checkpoint và nhật ký của đúng lần chạy đó vào nop-bai/ (theo cấu trúc ở Mục 3 của kế hoạch)."""
    run = ck.parent.parent  # runs/<màn>/<run>/checkpoints/step_*.pt
    (NOP / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ck, NOP / "checkpoints" / f"{level}.pt")
    dst = NOP / "runs" / level / run.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("config.yaml", "progress.csv", "evals.csv"):
        if (run / f).exists():
            shutil.copy2(run / f, dst / f)
        else:
            print(f"CẢNH BÁO: thiếu {run / f}")
    print(f"đã chép {ck} và nhật ký của {run} vào {NOP}/")


if __name__ == "__main__":
    main()
