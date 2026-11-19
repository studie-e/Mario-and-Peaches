# Kế hoạch đồ án Mario PPO

Nhóm: **Linh, Hiền, Cúc, Chi** · Thời gian: khoảng 17 ngày · 6 buổi họp
Mục tiêu: lấy tối đa A (30) + B (25) + C (15) + D (25) + E (5) + điểm thưởng (+10) = **110**

**Chiến lược:** cả 4 người cùng dồn sức theo thứ tự **1-1 → màn thưởng trong danh sách → màn ngoài danh sách**. Ở mỗi giai đoạn, 4 tài khoản Kaggle chạy 4 cấu hình hoặc 4 seed **cùng một màn**, nên có 4 cơ hội đạt cùng lúc. Các lần chạy đó đồng thời cho luôn số liệu cho thí nghiệm so sánh.

> Ngày họp ghi bên dưới là gợi ý, tính từ Thứ Sáu 20/11. Nếu hạn nộp khác thì giữ thứ tự và khoảng cách giữa các buổi, chỉ dời ngày.

---

## 0. Nguyên tắc chung (rút từ Mục 9 của đề)

| Thầy nhắc ở Mục 9 | Nhóm làm gì |
|---|---|
| Trình giả lập chạy trên CPU, mạng nhỏ nên GPU giúp ít | **Chỉ dùng phiên CPU của Kaggle** (4 lõi, không giới hạn giờ mỗi tuần) cho mọi việc. Không dùng GPU, nên không phải lo hết quota giữa chừng một lần chạy dài. Đặt `device=cpu` trong cấu hình. Colab chỉ dùng cho pytest, chấm checkpoint và vẽ đồ thị. |
| Tốc độ 100–500 bước/giây với 8–16 lõi | Kaggle chỉ có 4 lõi, nên **phải tự đo ở Buổi 2** rồi mới chốt ngân sách bước. Số bước mỗi phiên 12 giờ ≈ `bước/giây × 43 200 × 0,9`. |
| Chia việc, mỗi máy một cấu hình, chạy cùng lúc | Mỗi người một tài khoản Kaggle, cả 4 cùng chạy một màn với 4 cấu hình/seed khác nhau. **Không** lập tài khoản phụ (điều khoản Kaggle chỉ cho mỗi người một tài khoản). |
| Lần chạy bị ngắt thì chạy tiếp bằng `resume=<checkpoint>` | Phiên Kaggle tối đa 12 giờ, nên mọi lần chạy dài đều được chia thành nhiều phiên. Checkpoint được lưu thành Kaggle Dataset để phiên sau đọc lại. |
| Mỗi checkpoint (500k bước) kèm 5 lượt đánh giá và video để phát hiện sớm lần chạy hỏng | Giữ `checkpoint_every: 500_000`. Mỗi sáng người chạy xem `evals.csv` và video mới nhất, rồi áp dụng **quy tắc dừng sớm** ở mục 0.2. Khi một lần chạy hỏng, tài khoản đó chuyển ngay sang cấu hình khác. |

### 0.1 Quy tắc dùng chung
- Một repo git duy nhất. **Không sửa mã PPO lõi (`ppo.py`, `model.py`) sau Buổi 2.** Mẹo huấn luyện chỉ thêm vào wrapper mới và bật trong khối `if not eval_mode:` của `make_env`.
- **Không sửa** `evaluate.py` và phần `eval_mode` của `env.py`. Đầu vào mạng chỉ là khung hình.
- Mỗi lần chạy ghi một dòng vào **bảng theo dõi chung** (Google Sheet): người chạy, tài khoản, commit git, màn, cấu hình/khác biệt so với mặc định, seed, số bước, bước/giây, đường dẫn checkpoint, flag_rate theo lệnh Mục 4, ghi chú.
- Mọi kết luận "đạt" đều phải dựa trên **đúng lệnh ở Mục 4** (20 lượt, `--seed 2026`), không dựa trên `evals.csv` (vì file này chỉ có 5 lượt).
- Các lần chạy trong cùng một phép so sánh phải có **cùng `algo.total_steps`**, vì `anneal_lr: true` làm tốc độ học giảm dần theo tổng số bước. Một lần chạy 5M dừng ở mốc 2M không tương đương với một lần chạy 2M.

### 0.2 Quy tắc dừng sớm (quy ước của nhóm, không phải ngưỡng chuẩn)
Dừng và báo cả nhóm khi gặp một trong các tình huống sau:
- Sau 1M bước mà `x_pos` trung bình không tăng qua 2 checkpoint liên tiếp.
- Entropy giảm gần về 0 sớm trong khi `flag_rate` vẫn bằng 0 (chính sách đã "chốt" một lối chơi sai).
- `approx_kl` liên tục lớn bất thường so với các lần chạy khác, hoặc `clipfrac` tăng vọt, hoặc loss ra NaN.
- Video cho thấy Mario đứng yên hay lặp lại một hành động suốt nhiều checkpoint.

### 0.3 Lưu ý về điểm thưởng
Điểm thưởng **tối đa +10 và chỉ tính cho một màn ngoài danh sách**, nên làm thêm nhiều màn không được cộng dồn. Màn ngoài danh sách làm sau cùng, nên đây cũng là phần **đầu tiên bị cắt** nếu trễ tiến độ.

---

## 1. Vai trò của từng người

Việc **huấn luyện** thì cả 4 người cùng làm, theo bảng phân lần chạy của từng buổi họp ở mục 2. Mỗi người vẫn phụ trách riêng một phần mã, một mảng phân tích và một phần báo cáo:

| | Viết mã (đến Buổi 2) | Mảng phụ trách riêng | Phần báo cáo | Phần trình bày |
|---|---|---|---|---|
| **Hiền** | `wrappers.py` (FrameSkip, GrayResize, ScalarizeReward); dựng repo git; notebook Kaggle mẫu | Hạ tầng Kaggle, bảng theo dõi, **chấm checkpoint theo lệnh Mục 4**, đóng gói bài nộp | Cài đặt + bảng pytest; Thiết lập huấn luyện | Bài toán và môi trường, demo 1-1 |
| **Linh** | `model.py` (ActorCritic) | **Thí nghiệm so sánh** (thiết kế, tổng hợp kết quả); vẽ toàn bộ đồ thị | Đường cong học; Thí nghiệm so sánh | PPO và các chẩn đoán, thí nghiệm |
| **Cúc** | `compute_gae`, `value_loss`, `approx_kl_and_clipfrac` | **Học chuyển giao so với từ đầu** (so sánh trên màn thưởng) | Kết quả màn thưởng và màn ngoài danh sách | Màn thưởng |
| **Chi** | `clipped_policy_loss`, `ppo_loss`, `ppo_update` | **Phân tích thất bại**: xem video, tìm chỗ chết, đề xuất mẹo định hình phần thưởng | Phân tích thất bại (biểu đồ x cuối lượt) | Phân tích thất bại, demo màn thưởng |

- Mỗi người **review chéo** mã của người kế tiếp: Hiền → Linh → Cúc → Chi → Hiền.
- **Ai cũng phải tự giải thích được** GAE (khác nhau giữa terminated và truncated), mục tiêu cắt của PPO, approxKL và clipfrac, vì 5 phút hỏi đáp có thể hỏi bất kỳ ai.
- Hiền là người ghép báo cáo cuối và giữ trong giới hạn 8 trang. Linh là người ghép slide.

---

## 2. Sáu buổi họp

### Buổi 1: Khởi động (T6 20/11, khoảng 1 giờ)
**Phải chốt trong buổi:**
- [ ] Thống nhất bảng vai trò ở mục 1 và lịch 6 buổi họp.
- [ ] Cả 4 người có tài khoản Kaggle **đã xác minh số điện thoại** (bắt buộc để bật internet trong notebook, cần để clone repo và cài thư viện).
- [ ] Hiền tạo repo git và thêm cả nhóm. Tạo bảng theo dõi chung.
- [ ] Mỗi người đọc `de-bai.md`, `README.md` và docstring của các hàm mình phụ trách.
- [ ] Cử một người hỏi thầy: **màn ngoài danh sách có phải bất kỳ màn nào không nằm trong danh sách và khác 1-1 không** (ví dụ 1-3, 2-2), hay chỉ những màn như 4-2, 7-1, 8-1? Câu trả lời quyết định danh sách ứng viên cho điểm thưởng ở Buổi 5.

**Giao việc đến Buổi 2:**
| Người | Việc |
|---|---|
| Hiền | Viết `wrappers.py` cho đến khi `tests/test_wrappers.py` qua. Viết notebook Kaggle mẫu: chạy `nproc`, clone repo, `pip install -e .` rồi khởi động lại kernel (vì `numpy<2`), chạy `pytest`, train, ghi `runs/` vào `/kaggle/working`, lưu và đọc checkpoint qua Kaggle Dataset để `resume`. Notebook để Accelerator = **None** (phiên CPU) và chạy bằng "Save & Run All" để không bị ngắt khi không tương tác. |
| Linh | Viết `model.py` cho đến khi `tests/test_model.py` qua. Viết sẵn script vẽ đồ thị từ `progress.csv` và `evals.csv` (lợi tức, x_pos, flag_rate, entropy, approx_kl, clipfrac theo số bước), vẽ được nhiều lần chạy chồng lên nhau. |
| Cúc | Viết `compute_gae` cho đến khi `tests/test_gae.py` qua. Viết `value_loss` và `approx_kl_and_clipfrac`. |
| Chi | Viết `clipped_policy_loss`, `ppo_loss`, `ppo_update` cho đến khi `tests/test_ppo.py` qua. Viết sẵn script vẽ **biểu đồ x cuối lượt** từ file JSON của `evaluate.py`. |

---

### Buổi 2: Mã xong, đo tốc độ, cả 4 cùng chạy 1-1 (T2 23/11)
**Điều kiện để họp:** `pytest -q` báo **qua hết, kể cả `test_pipeline.py`**. Đây là mốc tuần 1 và là toàn bộ 30 điểm A.

**Phải hoàn thành trong buổi:**
- [ ] Review chéo xong, merge vào nhánh chính và **gắn tag commit `v1-code`**. Từ đây không sửa `ppo.py` và `model.py` nữa.
- [ ] Mỗi người đã chạy notebook mẫu trên tài khoản Kaggle của mình ít nhất một lần.
- [ ] **Đo tốc độ trên phiên CPU** (chạy trước buổi họp, mỗi lần khoảng 50k bước, `device=cpu`, mọi thứ khác theo cấu hình mặc định). Với 4 lõi, số môi trường song song và số luồng torch phải chia nhau CPU, nên đây là hai thông số cần dò:

| Người | `algo.num_envs` | `torch_threads` |
|---|---|---|
| Hiền | 16 | 2 |
| Linh | 8 | 2 |
| Cúc | 8 | 1 |
| Chi | 4 | 1 |

- [ ] Chốt **`num_envs` và `torch_threads`** cho mọi lần chạy về sau (tổ hợp có bước/giây cao nhất), rồi tính **số bước mỗi phiên 12 giờ**. Lưu ý: đổi `num_envs` thì số mẫu mỗi lần cập nhật (`num_envs × num_steps`) cũng đổi. Nếu chọn `num_envs` nhỏ, cân nhắc tăng `num_steps` để giữ khoảng 4096 mẫu, và ghi lựa chọn này vào báo cáo.
- [ ] Chốt **ngân sách 1-1: cùng một `total_steps` cho cả 4 lần chạy** (gợi ý 3M, điều chỉnh theo tốc độ đo được), để 4 lần chạy vừa là 4 cơ hội đạt 1-1, vừa là thí nghiệm so sánh hợp lệ.
- [ ] Linh ghi **giả thuyết** của thí nghiệm so sánh vào bảng theo dõi trước khi chạy. Mặc định so sánh **γ = 0,9 với γ = 0,99**, mọi thứ khác giữ nguyên.

**Giao việc đến Buổi 3: cả 4 cùng chạy 1-1** (commit `v1-code`, chạy ngay sau buổi họp):
| Người | Lần chạy 1-1 |
|---|---|
| Hiền | γ = 0,9 (mặc định), seed 1 |
| Linh | γ = 0,9 (mặc định), seed 2 |
| Cúc | γ = 0,99, seed 1 |
| Chi | γ = 0,99, seed 2 |

Mỗi sáng, người chạy kiểm tra `evals.csv` và video theo quy tắc ở mục 0.2. Trong lúc chờ: Cúc và Chi viết nháp phần mô tả GAE và hàm mất mát PPO cho báo cáo, Hiền viết nháp phần "Cài đặt".

---

### Buổi 3: Đọc kết quả 1-1, thăm dò màn thưởng (T5 26/11)
**Phải hoàn thành trong buổi:**
- [ ] Linh trình bày đồ thị cả 4 lần chạy vẽ chồng lên nhau. Cả nhóm nhận xét entropy, approxKL, clipfrac và chốt lần chạy nào tiếp tục, lần nào dừng.
- [ ] Hiền chấm checkpoint tốt nhất của **từng** lần chạy bằng **lệnh Mục 4**. Nếu có lần chạy ≥16/20 thì 1-1 coi như xong.
- [ ] Chọn **checkpoint 1-1 làm điểm xuất phát** cho học chuyển giao. Không cần đã đạt 0,8, miễn là tiến xa ổn định.

**Nếu chưa lần chạy nào tiến tốt**, Chi xem video và biểu đồ x cuối lượt để tìm chỗ chết, rồi cả 4 tài khoản chuyển sang 4 hướng cứu, mỗi hướng chỉ đổi một yếu tố so với lần chạy tốt nhất:
1. γ nào tốt hơn theo kết quả hiện có, seed mới.
2. Wrapper chỉ dùng khi huấn luyện: cắt lượt khi Mario không tiến trong N bước.
3. Wrapper chỉ dùng khi huấn luyện: thưởng thêm khi chạm cờ.
4. Tăng hệ số entropy nếu entropy rơi quá sớm.

**Giao việc đến Buổi 4:**
- Nếu 1-1 **chưa** đạt: cả 4 chạy 4 hướng cứu ở trên.
- Nếu 1-1 **đã** đạt (hoặc đã có checkpoint tiến xa để dùng làm điểm xuất phát): cả 4 cùng **thăm dò 8 màn trong danh sách** bằng học chuyển giao, mỗi màn 1M bước, mỗi người 2 màn:

| Người | Màn thăm dò |
|---|---|
| Hiền | 1-2, 1-4 |
| Linh | 2-1, 3-1 |
| Cúc | 3-3, 4-1 |
| Chi | 5-2, 6-1 |

Mỗi người ghi x_pos trung bình, flag_rate và một câu nhận xét từ video của từng màn vào bảng theo dõi. Nếu một lần chạy 1-1 còn đang tiến thì vẫn để nó chạy tiếp song song trên phiên khác, nếu tài khoản cho phép.

---

### Buổi 4: Chốt 1-1, chọn màn thưởng, cả 4 cùng chạy (CN 29/11)
**Phải hoàn thành trong buổi:**
- [ ] **1-1 đạt ≥16/20 theo lệnh Mục 4**, có JSON và video trong `nop-bai/`. Nếu vẫn chưa đạt thì 2 tài khoản tiếp tục cứu 1-1, 2 tài khoản bắt đầu màn thưởng.
- [ ] Chọn **màn thưởng trong danh sách** dựa trên số liệu thăm dò (màn có x_pos trung bình cao nhất và flag_rate > 0 nếu có), không chọn theo cảm tính.
- [ ] Linh trình bày kết quả thí nghiệm so sánh γ (đủ 2 seed): kết luận rõ, hoặc ghi nhận "chưa đủ bằng chứng" nếu chênh lệch giữa hai seed lớn hơn chênh lệch giữa hai cấu hình.
- [ ] Thống nhất dàn ý báo cáo và slide, gồm tên mục, người viết và số trang dự kiến.

**Giao việc đến Buổi 5: cả 4 cùng chạy màn thưởng** (ngân sách tối đa 10M bước mỗi lần chạy, tính cả phần đã thăm dò; chạy qua nhiều phiên bằng `resume`):
| Người | Lần chạy trên màn thưởng |
|---|---|
| Hiền | Học chuyển giao từ checkpoint 1-1, γ tốt nhất, seed 1 (tiếp tục từ lần thăm dò nếu đúng màn) |
| Linh | Học chuyển giao, cùng cấu hình, seed 2 |
| Cúc | **Huấn luyện từ đầu**, cùng cấu hình, seed 1 (để so sánh học chuyển giao với từ đầu) |
| Chi | Học chuyển giao + **một mẹo định hình phần thưởng** nhắm vào chỗ chết đã thấy khi thăm dò |

Viết báo cáo song song: Linh xong "Đường cong học" và "Thí nghiệm so sánh"; Hiền xong "Cài đặt" và "Thiết lập huấn luyện".

---

### Buổi 5: Chốt màn thưởng, chuyển sang màn ngoài danh sách (T4 02/12)
**Phải hoàn thành trong buổi:**
- [ ] Hiền chấm checkpoint tốt nhất của cả 4 lần chạy màn thưởng bằng **lệnh Mục 4** (đổi `--level`).
- [ ] Nếu **màn thưởng đã ≥16/20**: lưu JSON và video, rồi cả nhóm chuyển sang màn ngoài danh sách.
- [ ] Nếu **chưa đạt**: Chi trình bày biểu đồ x cuối lượt và video, chỉ ra chỗ tác tử hay chết. Cả 4 tài khoản tiếp tục cứu màn thưởng, các lần chạy không tiến được thay bằng mẹo định hình phần thưởng nhắm vào đúng chỗ đó. **Màn ngoài danh sách bị hoãn hoặc bỏ.**
- [ ] Cúc trình bày so sánh học chuyển giao với từ đầu (đường cong của Hiền/Linh so với Cúc).
- [ ] **Bản nháp báo cáo đầy đủ** (đủ mọi mục, có thể chưa đẹp). Mỗi hình có một đoạn nhận xét.

**Thứ tự ưu tiên nếu thiếu thời gian:** màn thưởng C (15đ) → báo cáo D (25đ) → màn ngoài danh sách (+10). Báo cáo không được hy sinh để chạy thêm.

**Giao việc đến Buổi 6 (khi màn thưởng đã đạt): cả 4 cùng làm màn ngoài danh sách.** Còn khoảng 4 ngày, nên bỏ qua thăm dò dài, mỗi người chạy thẳng một ứng viên bằng học chuyển giao từ checkpoint gần nhất (1-1 hoặc màn thưởng, chọn cái gần với kiểu màn đích hơn):
| Người | Lần chạy |
|---|---|
| Hiền | Ứng viên ngoài danh sách 1 |
| Linh | Ứng viên ngoài danh sách 2 |
| Cúc | Ứng viên ngoài danh sách 3 |
| Chi | Ứng viên 1 với seed khác hoặc kèm mẹo định hình phần thưởng |

Sau 1–2 phiên, dồn mọi tài khoản sang ứng viên đang tiến xa nhất. Danh sách ứng viên chốt theo câu trả lời của thầy ở Buổi 1.

Viết báo cáo song song: Chi xong "Phân tích thất bại"; Cúc xong phần kết quả màn thưởng (nêu rõ dùng học chuyển giao hay từ đầu, theo yêu cầu của đề); Linh hoàn thiện đồ thị và slide; Hiền ghép báo cáo vào khung 8 trang.

---

### Buổi 6: Chốt và tập trình bày (CN 06/12)
**Phải hoàn thành trong buổi:**
- [ ] Hiền chạy lại **đúng lệnh Mục 4** trên mọi checkpoint sẽ nộp, ghi lại flag_rate cuối cùng. Màn ngoài danh sách chỉ nộp nếu flag_rate > 0 (điểm thưởng tính theo tỉ lệ, nên đạt một phần vẫn có điểm).
- [ ] Kiểm tra checkpoint nộp **tái tạo được**: mỗi checkpoint gắn với `config.yaml`, `progress.csv`, `evals.csv` của đúng lần chạy đó, và với commit git tương ứng.
- [ ] Báo cáo PDF bản cuối (≤ 8 trang), có mục **Phân công** ghi đúng bảng ở mục 1 và các bảng phân lần chạy.
- [ ] **Tập trình bày 10 phút**, có bấm giờ. Mỗi người trình bày phần mình phụ trách. Sau đó hỏi đáp thử: mỗi người hỏi một người khác một câu về GAE/PPO.
- [ ] Nén `nhom-<số>.zip` và một người khác kiểm tra lại toàn bộ trước khi nộp.

---

## 3. Danh sách nộp bài (Hiền phụ trách)

```
nhom-<số>.zip
├── mario_ppo/, tests/, configs/, pyproject.toml, README.md   # toàn bộ mã
├── runs/
│   ├── 1-1/<run>/            config.yaml, progress.csv, evals.csv
│   ├── <màn-thưởng>/<run>/   config.yaml, progress.csv, evals.csv
│   └── <màn-ngoài>/<run>/    config.yaml, progress.csv, evals.csv
├── checkpoints/
│   ├── 1-1.pt
│   ├── <màn-thưởng>.pt
│   └── <màn-ngoài>.pt
├── danh-gia/
│   ├── ket-qua-1-1.json, 1-1.mp4
│   ├── ket-qua-<màn-thưởng>.json, <màn-thưởng>.mp4
│   └── ket-qua-<màn-ngoài>.json, <màn-ngoài>.mp4
├── bao-cao.pdf
└── trinh-chieu.pdf (hoặc .pptx)
```

Lệnh đánh giá (chỉ đổi `--ckpt`, `--level`, `--out`, `--video`):
```bash
python -m mario_ppo.evaluate --ckpt checkpoints/1-1.pt --level 1-1 --episodes 20 --seed 2026 --out danh-gia/ket-qua-1-1.json --video danh-gia/1-1.mp4
```

---

## 4. Bảng mục tiêu điểm

| Hạng mục | Điểm | Ai làm | Đạt khi | Chốt muộn nhất |
|---|---|---|---|---|
| A. Cài đặt | 30 | cả nhóm, mỗi người một phần mã | pytest qua hết | Buổi 2 |
| B. Màn 1-1 | 25 | **cả 4 cùng chạy** | ≥16/20 theo lệnh Mục 4 | Buổi 4 |
| C. Màn thưởng | 15 | **cả 4 cùng chạy** | ≥16/20 | Buổi 5 (cứu muộn nhất Buổi 6) |
| D. Báo cáo | 25 | cả nhóm, Hiền ghép | đủ 6 mục, mỗi hình có nhận xét, có thí nghiệm 2 seed | Buổi 6 |
| E. Trình bày | 5 | cả nhóm, Linh ghép slide | 10 phút, ai cũng trả lời được | Buổi 6 |
| Điểm thưởng | +10 | **cả 4 cùng chạy**, chỉ sau khi C đạt | ≥16/20 trên một màn ngoài danh sách | Buổi 6 |
