"""Đóng gói bài nộp nhom-<số>.zip theo Mục 7 của đề và Mục 3 của kế hoạch.

    python scripts/dong_goi.py 7            # tạo nhom-7.zip
    python scripts/dong_goi.py 7 --kiem-tra # chỉ kiểm tra, không nén

Mã nguồn lấy từ thư mục gốc repo; runs/, checkpoints/, danh-gia/, bao-cao.pdf, trinh-chieu.* lấy từ nop-bai/.
"""
import argparse
import zipfile
from pathlib import Path

MA_NGUON = ["mario_ppo", "tests", "configs", "scripts", "pyproject.toml", "README.md"]
NOP = Path("nop-bai")
BO_QUA = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def kiem_tra():
    loi, canh_bao = [], []
    for f in MA_NGUON:
        if not Path(f).exists():
            loi.append(f"thiếu mã nguồn {f}")
    for f in ("bao-cao.pdf",):
        if not (NOP / f).exists():
            loi.append(f"thiếu nop-bai/{f}")
    if not list(NOP.glob("trinh-chieu.*")):
        loi.append("thiếu nop-bai/trinh-chieu.pdf (hoặc .pptx)")

    levels = sorted(p.stem for p in (NOP / "checkpoints").glob("*.pt"))
    if "1-1" not in levels:
        loi.append("thiếu nop-bai/checkpoints/1-1.pt")
    if len(levels) < 2:
        loi.append("thiếu checkpoint màn thưởng trong nop-bai/checkpoints/")
    for lv in levels:
        for f in (f"danh-gia/ket-qua-{lv}.json", f"danh-gia/{lv}.mp4"):
            if not (NOP / f).exists():
                loi.append(f"{lv}: thiếu nop-bai/{f}")
        runs = list((NOP / "runs" / lv).glob("*/config.yaml"))
        if len(runs) != 1:
            loi.append(f"{lv}: cần đúng 1 lần chạy trong nop-bai/runs/{lv}/, đang có {len(runs)}")
        for r in runs:
            for f in ("progress.csv", "evals.csv"):
                if not (r.parent / f).exists():
                    loi.append(f"{lv}: thiếu {r.parent / f}")
    if len(levels) > 3:
        canh_bao.append(f"có {len(levels)} checkpoint ({', '.join(levels)}); kế hoạch chỉ nộp 1-1, màn thưởng, "
                        "và tối đa một màn ngoài danh sách")
    return levels, loi, canh_bao


def tep_ma_nguon():
    for f in MA_NGUON:
        p = Path(f)
        files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
        for x in files:
            if not BO_QUA.intersection(x.parts) and x.suffix != ".pyc":
                yield x, x.as_posix()


def tep_nop():
    for x in sorted(NOP.rglob("*")):
        if x.is_file() and x.name != "README.md":
            yield x, x.relative_to(NOP).as_posix()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("so_nhom")
    p.add_argument("--kiem-tra", action="store_true")
    args = p.parse_args(argv)

    levels, loi, canh_bao = kiem_tra()
    print(f"màn có checkpoint: {', '.join(levels) or '(chưa có)'}")
    for m in canh_bao:
        print("CẢNH BÁO:", m)
    for m in loi:
        print("LỖI:", m)
    if args.kiem_tra:
        return
    if loi:
        raise SystemExit("chưa đủ điều kiện nộp, sửa các lỗi trên rồi chạy lại")

    out = Path(f"nhom-{args.so_nhom}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in [*tep_ma_nguon(), *tep_nop()]:
            z.write(src, arc)
    print(f"đã tạo {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
