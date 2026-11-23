# Thí nghiệm của Linh: so sánh gamma trên màn 1-1

## Giả thuyết ghi trước khi chạy

Với cùng ngân sách bước và cùng các tham số khác, gamma 0,99 có thể giúp Mario
ưu tiên tiến về đích hơn gamma 0,9 do tính đến phần thưởng ở xa hơn. Đây là giả
thuyết để kiểm chứng, chưa phải kết quả. Gamma lớn hơn cũng có thể làm việc học
dao động hơn; cần xem cả hai seed trước khi kết luận.

## Thiết kế

| Người | Gamma | Seed |
|---|---:|---:|
| Hiền | 0,9 | 1 |
| Linh | 0,9 | 2 |
| Cúc | 0,99 | 1 |
| Chi | 0,99 | 2 |

Tất cả chạy màn 1-1 từ cùng commit `v1-code`, cùng `algo.total_steps` được nhóm
chốt sau khi đo tốc độ CPU. Giữ nguyên các tham số còn lại, kể cả kiến trúc mạng,
wrapper, số môi trường và số bước mỗi lần cập nhật. Mỗi người ghi commit, cấu
hình, số bước/giây và checkpoint vào bảng theo dõi chung.

Lệnh chạy của Linh sau khi nhóm chốt `total_steps` (thay `SO_BUOC`):

```bash
python -m mario_ppo.train level=1-1 device=cpu algo.gamma=0.9 seed=2 algo.total_steps=SO_BUOC
```

Nếu nhóm chọn `num_envs` và `num_steps` khác mặc định, thêm hai tham số đó vào
lệnh của **cả bốn** người. `torch_threads` lấy từ kết quả đo tốc độ.

## Đọc kết quả

Vẽ đồ thị từ các thư mục chạy:

```bash
python scripts/ve_do_thi.py RUN_HIEN RUN_LINH RUN_CUC RUN_CHI --label "gamma=.9 seed=1" --label "gamma=.9 seed=2" --label "gamma=.99 seed=1" --label "gamma=.99 seed=2" --output figures/so-sanh-gamma.png
```

So sánh `x_pos`, phần thưởng, `flag_rate` qua các mốc bước; dùng `entropy`,
`approx_kl`, `clipfrac` để phát hiện lần chạy bất thường. `evals.csv` chỉ gồm 5
lượt tại mỗi checkpoint nên dùng để theo dõi xu hướng. Kết quả đạt màn phải
được Hiền chấm riêng trên 20 lượt, seed 2026, theo lệnh đánh giá của kế hoạch.

Kết luận chỉ khi cả hai seed của một cấu hình có xu hướng nhất quán. Nếu chênh
lệch giữa các seed lớn hơn chênh lệch giữa hai gamma, ghi “chưa đủ bằng chứng”.
Chưa có dữ liệu huấn luyện, nên chưa thể điền kết quả hoặc nhận xét đồ thị.

## Nháp nội dung báo cáo và trình bày của Linh

Mạng ActorCritic nhận bốn khung hình xám 84 × 84 được xếp theo kênh. Ba lớp
tích chập trích đặc trưng ảnh, sau đó một lớp tuyến tính tạo biểu diễn chung.
Đầu actor xuất logits cho từng hành động; đầu critic xuất một giá trị trạng
thái. Điểm ảnh được chia 255 trước khi vào mạng. Đầu actor khởi tạo nhỏ để
chính sách ban đầu gần đều và có cơ hội khám phá nhiều hành động.

Khi có dữ liệu, phần “Đường cong học” cần hình so sánh các seed theo số bước,
nhận xét rõ giai đoạn Mario bắt đầu tiến xa, giai đoạn tỷ lệ chạm cờ thay đổi,
và có hay không dấu hiệu bất ổn. Entropy thấp sớm có thể là chính sách hết khám
phá; approx KL hoặc clipfrac cao bất thường gợi ý mỗi cập nhật đang thay đổi
chính sách quá nhiều. Đây là dấu hiệu cần kiểm tra cùng video, không tự nó
chứng minh một lần chạy thất bại.

Phần “Thí nghiệm so sánh” cần bảng bốn lần chạy (gamma, seed, số bước, tốc độ,
checkpoint, kết quả chấm 20 lượt), hình vẽ chồng, rồi kết luận dựa trên sự
nhất quán của hai seed. Chỉ điền số liệu sau khi có log và kết quả đánh giá.

Gợi ý ba slide của Linh: (1) mạng actor–critic và ý nghĩa PPO; (2) đường cong
huấn luyện bốn lần chạy với dấu hiệu từ entropy/KL/clipfrac; (3) so sánh gamma
và kết luận. Các slide kết quả chỉ hoàn thiện khi nhóm cung cấp log thực tế.
