# nop-bai/: bản nộp cuối (Hiền phụ trách)

Thư mục này chứa các tệp sẽ nộp, ngoài mã nguồn. Mã nguồn được `scripts/dong_goi.py` lấy từ gốc repo.

```
nop-bai/
├── runs/<màn>/<run>/     config.yaml, progress.csv, evals.csv   (đúng lần chạy sinh ra checkpoint nộp)
├── checkpoints/<màn>.pt                                         (không đưa lên git, xem .gitignore)
├── danh-gia/ket-qua-<màn>.json, <màn>.mp4                       (sinh bằng lệnh Mục 4)
├── bao-cao.pdf
└── trinh-chieu.pdf (hoặc .pptx)
```

Cách đưa một checkpoint vào đây (chấm đúng lệnh Mục 4, ghi video, chép nhật ký của lần chạy):

```bash
python scripts/cham_diem.py --level 1-1 --nop runs/1-1/<run>/checkpoints/step_003000000.pt
```

Kiểm tra và nén:

```bash
python scripts/dong_goi.py <số-nhóm> --kiem-tra
python scripts/dong_goi.py <số-nhóm>
```

Nếu checkpoint màn thưởng được học chuyển giao từ checkpoint 1-1, ghi rõ trong báo cáo checkpoint xuất phát
(`scripts/chuyen_giao.py` lưu đường dẫn nguồn trong khóa `nguon` của tệp .pt).
