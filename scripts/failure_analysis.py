"""Vẽ biểu đồ vị trí x cuối mỗi lượt từ tệp JSON do `mario_ppo.evaluate` sinh (phục vụ phần "Phân tích thất bại").

Mỗi tệp JSON là một hàng biểu đồ: histogram của x_pos cuối lượt (20 lượt), vạch đứt là x trung bình,
vạch chấm là vị trí cờ (nếu cho --flag-x), các chấm nhỏ bên dưới là từng lượt. Lượt dùng hết --max-steps bước
(tác tử không chết cũng không chạm cờ, thường là đứng yên hoặc kẹt) được tô màu khác.

    python scripts/failure_analysis.py danh-gia/ket-qua-1-1.json --out hinh/x-cuoi-1-1.png
    python scripts/failure_analysis.py a.json b.json c.json --out so-sanh.png --bin-width 80 --flag-x 3161

Lưu ý: JSON của evaluate.py chỉ lưu flag_rate chung, không lưu từng lượt có chạm cờ hay không, nên script không
tô riêng lượt chạm cờ. Dùng --flag-x để thấy các lượt dừng gần vị trí cờ.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C_BAR, C_STUCK, C_MEAN, C_FLAG = "#4c78a8", "#e45756", "#222222", "#54a24b"


def load(path):
    res = json.loads(Path(path).read_text())
    for key in ("x_pos", "steps"):
        if key not in res:
            raise SystemExit(f"{path}: thiếu khóa '{key}', đây có phải JSON do mario_ppo.evaluate sinh không?")
    return res


def label_of(path, res):
    name = Path(path).stem
    step = res.get("global_step")
    extra = f", {step / 1e6:.2f}M bước" if step else ""
    return f"{name} (màn {res.get('level', '?')}{extra})"


def draw(ax, res, title, bin_width, flag_x, max_steps, xmax):
    x = np.asarray(res["x_pos"], dtype=float)
    steps = np.asarray(res["steps"])
    stuck = steps >= max_steps
    edges = np.arange(0, xmax + bin_width, bin_width)

    ax.hist(x[~stuck], bins=edges, color=C_BAR, label="kết thúc lượt (chết hoặc chạm cờ)")
    if stuck.any():
        ax.hist(x[stuck], bins=edges, color=C_STUCK, bottom=np.histogram(x[~stuck], bins=edges)[0],
                label=f"hết {max_steps} bước")
    ax.axvline(x.mean(), color=C_MEAN, ls="--", lw=1.2, label=f"x trung bình {x.mean():.0f}")
    if flag_x is not None:
        ax.axvline(flag_x, color=C_FLAG, ls=":", lw=1.5, label=f"cờ (x = {flag_x:.0f})")
    # từng lượt, rải theo chiều dọc cho khỏi chồng nhau
    rng = np.random.default_rng(0)
    ax.scatter(x, -0.35 - 0.25 * rng.random(len(x)), s=10, c=np.where(stuck, C_STUCK, C_BAR), clip_on=False)

    n = len(x)
    ax.set_title(f"{title}: flag_rate {res.get('flag_rate', float('nan')):.2f} ({round(res.get('flag_rate', 0) * n)}/{n} lượt)",
                 fontsize=10, loc="left")
    ax.set_ylabel("số lượt")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json", nargs="+", help="một hay nhiều tệp JSON của evaluate.py")
    p.add_argument("--out", default="x-cuoi-luot.png")
    p.add_argument("--bin-width", type=float, default=100, help="độ rộng mỗi cột histogram (đơn vị x)")
    p.add_argument("--flag-x", type=float, default=None, help="vị trí x của cờ, vẽ vạch xanh nếu có")
    p.add_argument("--max-steps", type=int, default=3000, help="giới hạn bước mỗi lượt đã dùng khi đánh giá")
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args(argv)

    results = [(path, load(path)) for path in args.json]
    xmax = max(max(r["x_pos"]) for _, r in results)
    if args.flag_x is not None:
        xmax = max(xmax, args.flag_x)
    xmax = (int(xmax // args.bin_width) + 1) * args.bin_width

    fig, axes = plt.subplots(len(results), 1, figsize=(8, 2.6 * len(results) + 0.8), sharex=True, squeeze=False)
    for ax, (path, res) in zip(axes[:, 0], results):
        draw(ax, res, label_of(path, res), args.bin_width, args.flag_x, args.max_steps, xmax)
    axes[-1, 0].set_xlabel("vị trí x cuối lượt")
    axes[-1, 0].set_xlim(0, xmax)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"đã ghi {args.out}")


if __name__ == "__main__":
    main()
