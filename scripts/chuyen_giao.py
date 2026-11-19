"""Tạo checkpoint xuất phát cho học chuyển giao: giữ trọng số, đặt lại số bước và optimizer.

`resume=<ckpt>` của train.py nạp cả optimizer lẫn global_step. Nếu dùng thẳng checkpoint 1-1 (ví dụ ở bước 3M)
thì lần chạy màn thưởng bắt đầu từ bước 3M: ngân sách 10M bị tính sai và lr đã bị anneal. Script này đưa
global_step về 0 và tạo optimizer Adam mới, để `total_steps` của màn mới tính từ đầu.

    python scripts/chuyen_giao.py runs/1-1/<run>/checkpoints/step_003000000.pt checkpoints/xuat-phat-1-1.pt
    python -m mario_ppo.train level=1-2 resume=checkpoints/xuat-phat-1-1.pt algo.total_steps=10000000

Lưu ý: lần chạy mới phải giữ cùng env.actions và env.frame_stack với checkpoint nguồn (nếu khác, nạp trọng số lỗi).
"""
import argparse
from pathlib import Path

import torch

from mario_ppo.evaluate import load_model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="checkpoint nguồn (.pt)")
    p.add_argument("dst", help="checkpoint xuất phát sẽ tạo (.pt)")
    args = p.parse_args(argv)

    model, ck = load_model(args.src)
    a = ck["config"]["algo"]
    opt = torch.optim.Adam(model.parameters(), lr=a["lr"], eps=a["adam_eps"])
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "global_step": 0,
                "config": ck["config"], "nguon": str(args.src), "nguon_global_step": int(ck["global_step"])},
               args.dst)
    print(f"{args.dst}: trọng số từ {args.src} (bước {ck['global_step']}, màn {ck['config']['level']}), "
          f"global_step = 0, optimizer mới")


if __name__ == "__main__":
    main()
