# Ghi chú triển khai DQN cho FlappyBird

Ghi chú này đi kèm [Problem_Definition.md](Problem_Definition.md) và chứa các chi tiết cần khi lập trình agent. Số liệu ghi "đo được" lấy từ việc chạy thử thư viện trên máy phát triển của dự án; các số còn lại tính từ mã nguồn của thư viện.

## 1. Dùng thư viện

- Phiên bản: flappy-bird-gymnasium 0.4.0, gymnasium 1.3.0 và Python 3.11, cài trong conda env `flappybird`. Mạng nơ-ron viết bằng TensorFlow 2.21 và Keras 3.15.1, cũng có sẵn trong env này.
- Tạo môi trường bằng `gymnasium.make("FlappyBird-v0", use_lidar=False)`. Mặc định của thư viện là chế độ LIDAR (180 tia đo khoảng cách), nên phải đặt `use_lidar=False`.
- `gymnasium.make` chỉ bọc thêm hai lớp kiểm tra `OrderEnforcing` và `PassiveEnvChecker`. Môi trường không khai báo `max_episode_steps`, nên không có `TimeLimit` cắt ván theo số bước.
- Mỗi lần gọi `step()` là một khung hình và trả về `obs, reward, terminated, truncated, info`. `terminated` là `True` khi chim va chạm. `truncated` chỉ là `True` khi đã đặt `score_limit` và điểm đạt giới hạn; lúc đó chim chưa chết. `info["score"]` là số ống đã vượt, chỉ dùng để chấm điểm.
- Sau khi `terminated`, phải tự gọi `reset()`. Thư viện không chặn `step()` tiếp theo và vẫn chạy với con chim đã va chạm. Đo được: trong 80 lần gọi `step()` sau khi chim đâm vào ống, có 30 lần trả về `terminated = False`. Các transition đó không được đưa vào agent.
- Docstring của thư viện đã lỗi thời. Nó ghi observation gồm 2 giá trị và `step()` trả về 4 giá trị, trong khi thực tế là 12 và 5.
- Huấn luyện với `render_mode=None`. Chế độ `"human"` bị khóa ở 30 khung hình/giây, nên 1 triệu khung hình mất khoảng 9,3 giờ. Không vẽ hình thì thư viện chạy khoảng 150.000 bước/giây, còn chế độ LIDAR chỉ 1.152 bước/giây.
- Theo quy ước kiến trúc của dự án ([CLAUDE.md](CLAUDE.md): tách game, môi trường và AI), chỉ module môi trường được import `gymnasium`. Agent, replay buffer và Q-Network chỉ làm việc với mảng số, nên kiểm thử được bằng dữ liệu giả.

## 2. Observation

Observation gồm 12 số `float64`, mặc định đã được chia tỉ lệ (`normalize_obs=True`) về khoảng từ −1 đến 1. Cần đổi sang `float32` trước khi đưa vào mạng. Tọa độ tính bằng pixel trên màn hình 288 × 512, gốc ở góc trên bên trái, trục y hướng xuống.

| Chỉ số | Ý nghĩa | Chia cho | Giá trị trước khi chia |
|---|---|---|---|
| 0, 3, 6 | x của mép trái ba ống, xếp theo x tăng dần | 288 | −52 đến 288 |
| 1, 4, 7 | y của mép trên khe (mép dưới ống trên) | 512 | 100 đến 170 (0 khi bị che) |
| 2, 5, 8 | y của mép dưới khe (mép trên ống dưới) | 512 | 200 đến 270 (512 khi bị che) |
| 9 | y của cạnh trên con chim | 512 | −92 đến 380 |
| 10 | Vận tốc dọc `vy`, âm là đi lên | 10 | −9 đến 10 |
| 11 | Góc nghiêng `rot` | 90 | −90 đến 45 |

- Ống có x > 288, tức chưa vào màn hình, bị thay bằng `(288, 0, 512)`, hay `(1, 0, 1)` sau khi chia. Ống mới được tạo ở x = 397,6 nên bị che trong khoảng 28 khung hình đầu. Đo được: bộ thứ ba (chỉ số 6, 7, 8) bị che 75% thời gian; bộ thứ hai (chỉ số 3, 4, 5) chỉ bị che trong 36 khung hình đầu mỗi ván, chiếm 2,7% số khung hình.
- Ống cần vượt là bộ thứ nhất nếu x₀ + 52 − 57 > 0, tức mép phải của ống đó chưa qua mép trái của chim (x₀ tính bằng pixel, tức giá trị ở chỉ số 0 nhân 288); ngược lại là bộ thứ hai. Ống chim vừa bay qua ở lại bộ thứ nhất thêm khoảng 14 khung hình. Đo bằng luật viết tay: ống cần vượt ở bộ thứ nhất trong 63,7% số khung hình và ở bộ thứ hai trong 36,3%.
- `rot` chỉ dùng để vẽ và không ảnh hưởng vật lý. Khi `vy` < 10 thì `rot = 18 − 3·vy`. Có thể bỏ nó khỏi đầu vào.

## 3. Luật vật lý và reward

Chim luôn ở x = 57, rộng 34 và cao 24 pixel; ống rộng 52 pixel.

```
Nếu FLAP và y > −48:  vy' = −9
Ngược lại:            vy' = min(vy + 1, 10)

y'     = y + min(vy', 404.48 − y − 24)
x_ống' = x_ống − 4
Nếu x_ống' < −52: ống được đặt lại ở x = 397.6, với khe mới chọn ngẫu nhiên
```

404,48 là độ cao mặt đất (0,79 × 512), và số hạng `min` giữ chim không rơi xuyên qua đất. FLAP bị bỏ qua khi chim đã cao hơn mép trên màn hình 48 pixel (y ≤ −48). Mép trên của khe nhận một trong 8 giá trị 100, 110, …, 170 với xác suất bằng nhau, và khe cao 100 pixel.

| Đại lượng | Giá trị |
|---|---|
| Độ cao tăng sau một lần FLAP | 45 pixel |
| Khoảng trống dọc của chim trong khe | 76 pixel |
| Số khung hình chim chồng lên một ống theo chiều ngang, tức có thể va chạm | 21 |
| Tốc độ ống | 4 pixel mỗi khung hình |
| Khoảng cách giữa hai ống liên tiếp | 144 hoặc 164 pixel, tức 36 hoặc 41 khung hình |
| Chu kỳ lặp của ba ống | 113 khung hình, trung bình khoảng 38 khung hình mỗi ống |

- Lúc bắt đầu, chim ở y = 244 với `vy` = −9, và ba ống ở x = 288, 432 và 576 với khe ngẫu nhiên.
- Chim chạm đất khi y ≥ 379,48, tức đáy chim chạm mặt đất. Ống trên kéo dài lên tới y = −150 hoặc cao hơn, còn chim không lên cao hơn y = −92, nên không thể bay vòng qua phía trên ống.
- Reward được gán theo thứ tự, giá trị sau ghi đè giá trị trước: +1 nếu đường giữa chim vượt qua đường giữa ống, nếu không thì +0,1; sau đó −0,5 nếu y' < 0; sau đó −1 nếu va chạm.
- Reward +1 đến khi chim mới đi được nửa ống: đường giữa chim ở x = 74, đường giữa ống ở x_ống + 26. Chim vẫn có thể đâm vào nửa sau của ống trong 9 khung hình tính từ lúc đó, kể cả khung hình nhận +1 (10 khung hình với ba ống đầu tiên của ván).
- Đo bằng luật viết tay (200 ván, 267.492 khung hình): 97,37% số bước nhận +0,1, 2,56% nhận +1, và 0,075% là va chạm.

## 4. Số liệu đứng sau lập luận

### Mốc so sánh

| Cách chơi | Số ván | Độ dài ván trung bình (khung hình) | Kết quả |
|---|---|---|---|
| Ngẫu nhiên, P(FLAP) = 0,5 | 1.000 | 50,0 | Cả 1.000 ván chết ở khung hình 50, điểm 0 |
| Ngẫu nhiên, P(FLAP) = 0,1 | 1.000 | 50,5 | 6,3% số ván vượt được ít nhất 1 ống |
| Ngẫu nhiên, P(FLAP) = 0,08 | 1.000 | 50,1 | 6,9% |
| Ngẫu nhiên, P(FLAP) = 0,03 | 1.000 | 42,4 | 2,6% |
| Luật viết tay: vỗ cánh khi và chỉ khi, nếu không vỗ, đáy chim ở khung hình sau sẽ xuống thấp hơn một vạch đặt cao hơn mép dưới khe của ống cần vượt 12 pixel | 200 | 1.337 | Trung bình 34,2 ống, trung vị 25, từ 4 đến 160 |

Với P(FLAP) = 0,5, chim đâm vào ống trên của ống đầu tiên đúng lúc ống này tới chỗ chim ở khung hình 50, trong khi lần ghi điểm sớm nhất cần 61 khung hình. Luật viết tay vỗ cánh ở 5,8% số khung hình, trung vị 18 khung hình giữa hai lần vỗ.

### Cài đặt luật viết tay

Code: [Agent/hand_rule.py](Agent/hand_rule.py), hàm `hand_rule(obs)`. Luật chỉ đọc observation (mục 2) và dùng các hằng số vật lý ở mục 3. Ở mỗi khung hình:

```
# 1. Ống cần vượt (mục 2): bộ thứ nhất nếu mép phải của nó chưa qua mép trái chim
x₀          = obs[0] · 288
gap_bottom  = (obs[2] nếu x₀ + 52 > 57, ngược lại obs[5]) · 512

# 2. Vị trí chim ở khung hình sau nếu IDLE (mục 3)
y           = obs[9] · 512
vy'         = min(obs[10] · 10 + 1, 10)
y'          = y + min(vy', 404.48 − y − 24)

# 3. Quyết định
FLAP  nếu  y' + 24 > gap_bottom − 12,   ngược lại IDLE
```

- `y' + 24` là đáy chim, vì chim cao 24 pixel. Vạch `gap_bottom − 12` cách mép dưới khe 12 pixel, khoảng nửa thân chim.
- Luật không dùng `rot`, ống thứ ba và mép trên khe.
- Kiểm tra lại bằng code trên (200 ván, `score_limit` 1.000): seed 0, 1, 2 cho trung bình 35,0 / 35,2 / 37,0 ống, trung vị 23–27, thấp nhất 4. Với seed 0, tỷ lệ vỗ cánh là 5,81% trên 273.409 khung hình. Bảng trên ghi 34,2 vì seed của lần đo đó không được lưu. Chênh lệch nằm trong mức dao động giữa các seed, nên các tài liệu khác vẫn dùng mốc 34,2.
- Chạy lại: `python Agent/hand_rule.py --episodes 200 --seed 0`.

### Cỡ bài toán

Chim đạt được khoảng 8.500 cặp (y, `vy`) khác nhau, vị trí ba ống lặp lại sau 113 khung hình, và hai ống đang hiện có 8 × 8 = 64 tổ hợp khe. Tích là 8.500 × 113 × 64 ≈ 6 × 10^7. Khoảng 1/4 thời gian cả ba ống cùng hiện, khi đó có 8³ = 512 tổ hợp khe, nên tổng cộng khoảng 1,7 × 10^8 trạng thái.

Nếu biết T và R, quy hoạch động vẫn nặng. T rất thưa, mỗi cặp (s, a) chỉ dẫn tới 1 hoặc 8 trạng thái, nên một vòng quét tốn cỡ S·A phép tính thay vì S²A như trường hợp tổng quát, với S và A là số trạng thái và số hành động. Nhưng vẫn phải giữ một bảng cỡ 10^8 giá trị và quét hàng trăm vòng để giá trị lan đủ xa với γ = 0,99 (xem phần Target Network ở mục 5).

Nhìn trước một ống (38 bước) có 2^38 ≈ 2,7 × 10^11 chuỗi hành động. Gộp các chuỗi dẫn tới cùng một cặp (y, `vy`) thì còn 56.000 đến 86.000 lần gọi luật chơi cho mỗi quyết định (đo được).

### Kiểm tra "có thể chơi mãi"

Mỗi ống chỉ nguy hiểm trong 21 khung hình, khi chim chồng lên ống theo chiều ngang. Trong 21 khung hình đó, y của chim phải nằm trong khoảng từ mép trên khe đến mép trên khe + 76, tức cả thân chim lọt trong khe. Khe của ống kế tiếp hiện ra trong observation khoảng 49 khung hình trước khi ống đó trở nên nguy hiểm.

Phép kiểm tra dùng một bản luật vật lý chép từ mã nguồn, khớp hoàn toàn với môi trường trên 224.829 khung hình thử. Trên bản này, tính được một tập các cặp (y, `vy`) lúc chim vừa ra khỏi một ống, có tính chất sau: từ mỗi cặp trong tập, dù khe kế tiếp ở đâu, chim vẫn qua được ống kế tiếp và ra khỏi nó với một cặp cũng thuộc tập. Từ trạng thái bắt đầu cũng vào được tập này. Do đó với mọi chuỗi khe luôn có cách vỗ cánh để không bao giờ va chạm, không cần bay lên trên mép màn hình, và chỉ dùng thông tin có trong observation. Kết quả này còn dư: thu hẹp vùng an toàn 12 pixel ở mỗi phía vẫn có cách chơi không thua.

### Chọn γ và giá trị của việc sống

| γ | Trọng số sau 38 khung hình | Effective Horizon 1/(1−γ) | Giá trị khi sống mãi |
|---|---|---|---|
| 0,9 | 0,018 | 10 khung hình | 1,0 đến 1,9 |
| 0,99 | 0,68 | 100 khung hình, khoảng 2,6 ống | 11,9 đến 12,9, trung bình 12,4 |
| 0,999 | 0,96 | 1.000 khung hình | khoảng 124 |

Effective Horizon là số bước mà reward tương lai còn ảnh hưởng đáng kể; với |r| ≤ 1, tổng có chiết khấu luôn nhỏ hơn 1/(1−γ). Cột cuối dao động vì phụ thuộc vào việc ống kế tiếp còn xa hay gần, và không cách chơi nào nhận được nhiều hơn. Vì có cách chơi không bao giờ thua, đây cũng là giá trị tối ưu V\* của các trạng thái an toàn, tức những trạng thái mà từ đó chim vẫn tránh được va chạm mãi mãi mà không phải bay lên trên mép màn hình.

Với γ = 0,99, chết làm mất khoảng 12,4 − (−1) ≈ 13,4, trong đó reward −1 chỉ chiếm 7,5%.

### Đánh giá quá cao

Khi hai giá trị thật bằng nhau và mỗi ước lượng có sai số độc lập với độ lệch chuẩn σ, lấy max cho kết quả lệch lên trung bình σ/√π ≈ 0,56σ. Double DQN được dùng để giảm độ lệch này.

### ε

Với hai hành động, ε-greedy chọn hành động không tốt nhất với xác suất ε/2. Xác suất qua một ống (38 quyết định) mà không lệch lần nào là (1 − ε/2)^38:

| ε | Xác suất qua một ống không lệch lần nào |
|---|---|
| 0,1 | 14% |
| 0,05 | 38% |
| 0,01 | 83% |
| 0,001 | 98% |

Không phải lần lệch nào cũng gây chết. Ở vùng trống, sai một khung hình thường sửa được ngay sau đó; nguy hiểm tập trung lúc chim đang trong ống hoặc sát mép khe. Bảng trên vì vậy là cận dưới của xác suất qua ống an toàn.

## 5. Cấu hình khởi đầu

Các giá trị dưới đây được chọn bằng lập luận. Cấu hình này đã được chạy 3 lần trên Colab (dừng ở 1, 5 và 10 triệu khung hình) và học được; kết quả ở [README.md](README.md) mục 5–6, đối chiếu với lập luận ở cuối mục này. Mới có một cấu hình nên các giá trị chưa được so sánh với lựa chọn khác.

| Tham số | Giá trị |
|---|---|
| γ | 0,99 |
| Mạng | MLP: 12 đầu vào, hai lớp ẩn 256 nơ-ron dùng ReLU, đầu ra Dueling; khoảng 70 nghìn tham số |
| Optimizer | Adam, learning rate 0,0001 |
| Loss | MSE, tức trung bình của (Target − Q)² |
| Experience Replay | 100.000 transition; bắt đầu học khi đã có 10.000 |
| Batch | 64 transition |
| Tần suất cập nhật | 1 lần mỗi 4 khung hình |
| Đồng bộ Target Network | Mỗi 1.000 lần cập nhật, tức 4.000 khung hình |
| ε | Giảm đều từ 1,0 về 0,01 trong 100.000 khung hình đầu, sau đó giữ nguyên; có thể giảm tiếp về 0,001 khi agent đã vượt được nhiều ống |
| Xác suất FLAP khi chọn ngẫu nhiên | 0,1 trong notebook Colab (`random_flap_prob`); 0,5 trong các notebook `01` đến `06` |
| Tổng thời lượng | Ít nhất 1 triệu khung hình theo lập luận dưới đây; đo được cần khoảng 2 triệu để Q ổn định (xem cuối mục) |

Lý do của các giá trị trên:

- Khoảng 70 nghìn tham số là đủ cho 12 đầu vào. Mạng lớn hơn nhiều dễ học thuộc lòng những đoạn dữ liệu đang chiếm replay.
- 100.000 transition chiếm khoảng 11 MB ở dạng `float32` và chứa khoảng 2.650 ống khi agent đã sống lâu. Chờ đủ 10.000 transition (tương đương 200 ván ngẫu nhiên) để các batch đầu tiên đủ đa dạng.
- Chỉ cập nhật mỗi 4 khung hình vì các khung hình liền nhau gần trùng nhau, cập nhật ở mọi khung hình thêm được ít thông tin mới.
- Theo cách nhìn đơn giản, mỗi chu kỳ đồng bộ Target Network giống một vòng Value Iteration: giá trị lan ngược thêm khoảng một bước. Sau k lần đồng bộ, phần chưa lan tới còn cỡ γ^k, và muốn γ^k ≤ 0,1 với γ = 0,99 thì k ≈ 230. 1 triệu khung hình cho khoảng 250 lần đồng bộ, vừa đủ mức này, nên đó là mức tối thiểu.
- Nếu loss dao động mạnh, có thể thay MSE bằng Huber Loss. Huber Loss giống MSE khi sai số nhỏ nhưng chỉ tăng tuyến tính khi sai số lớn, nên vài Target lệch nhiều không kéo mạng đi quá xa.
- Khi thử ngẫu nhiên, FLAP được chọn với xác suất 50%, gấp khoảng 8,6 lần tần suất vỗ cánh của luật viết tay (5,8%). Có thể cho hành động ngẫu nhiên là FLAP với xác suất nhỏ hơn, ví dụ 0,1: khi đó 6,3% số ván ngẫu nhiên vượt được ống đầu tiên, so với 0% khi xác suất là 0,5. Vì Q-Learning là off-policy, Target không cần đổi. Notebook Colab `Code/FlappyBird_Colab.ipynb` dùng giá trị 0,1 (tham số `random_flap_prob`); các notebook `01` đến `06` vẫn chọn đều 0,5.

Đối chiếu với 3 lần chạy trên Colab (số liệu từ `Results/dqn_run*/history.jsonl`, phân tích ở `Results_Analysis.ipynb`):

- Mean max Q trên batch đạt 10,0–10,7 ở 1 triệu khung hình (250 lần đồng bộ) và chỉ vào vùng 12 từ khoảng 2 triệu (500 lần đồng bộ), rồi ổn định ở 12,2–12,5, không vượt 12,9. Lập luận "250 lần đồng bộ là vừa đủ" vì vậy đánh giá thấp thời lượng cần thiết khoảng 2 lần.
- Điểm đánh giá bắt đầu tăng ở khoảng 350.000–500.000 khung hình. Checkpoint tốt nhất đạt 908,8 và 910,7 ống mỗi ván (100 ván, giới hạn 1.000), lấy ở 850.000 và 3.250.000 khung hình.
- Cuối run, điểm huấn luyện chỉ khoảng 33–39 ống mỗi ván, trong khi đánh giá với ε = 0 đạt 150–200 ở giới hạn 200. Với ε = 0,01, vài hành động ngẫu nhiên trong một ván dài đã đủ làm chim chết; việc giảm ε về 0,001 chưa được thử.
- Ba lần chạy cùng cấu hình chênh nhau nhiều ở cùng mốc: trung bình 10 lần đánh giá gần nhất tại 1 triệu khung hình là 85,7, 144,6 và 90,9. Muốn so sánh hai cấu hình cần nhiều lần chạy cho mỗi cấu hình.

Mạng được cập nhật theo công thức sau. Ký hiệu Q(s, a; w) là Q do mạng với trọng số w tính ra. Loss là ½(Target − Q(s,a;w))², và Gradient Descent dịch w ngược hướng gradient, coi Target là hằng số:

```
δ = Target − Q(s,a;w)            (TD Error)
w ← w + α · δ · ∇_w Q(s,a;w)
```

Khi Q là tổng có trọng số của các đặc trưng f(s, a), ∇_w Q = f(s, a), và công thức trở thành đúng cập nhật của Approximate Q-Learning: w ← w + α·δ·f(s,a). Mạng nơ-ron chỉ thay vector đặc trưng cố định bằng gradient của mạng. Hệ số ½ chỉ để gradient gọn hơn; bỏ nó tương đương nhân đôi α. Target chỉ thật sự là hằng số khi không phụ thuộc vào w, và Target Network giữ điều này đúng trong 1.000 lần cập nhật giữa hai lần đồng bộ.

Đầu ra Dueling:

```
h      = ReLU(W₂ · ReLU(W₁ · s))
V(s)   = w_V · h
A(s,a) = W_A · h
Q(s,a) = V(s) + A(s,a) − (trung bình theo a' của A(s,a'))
```

W₁, W₂, w_V và W_A là các ma trận trọng số; các hệ số bias được bỏ qua cho gọn. Phép trừ trung bình làm V và A được xác định duy nhất, vì nếu không, cộng c vào V và trừ c khỏi A vẫn cho cùng một Q.

## 6. Vòng huấn luyện và đánh giá

Một vòng học gồm năm việc:

1. Chọn hành động bằng ε-greedy.
2. Chơi một khung hình và lưu transition, kèm cờ `terminated`, vào Experience Replay. Nếu ván kết thúc, gọi `reset()`.
3. Cứ 4 khung hình một lần, lấy ngẫu nhiên 64 transition, tính Target cho từng transition, rồi cập nhật mạng một bước để giảm (Target − Q)².
4. Cứ 1.000 lần cập nhật, chép trọng số mạng chính sang Target Network.
5. Cứ 25.000 khung hình, đánh giá trên một môi trường riêng và lưu lại phiên bản tốt nhất.

Target theo Double DQN:

```
a*     = argmax_a' Q(s', a'; w)        mạng chính chọn hành động
Target = r                             nếu terminated
Target = r + γ · Q(s', a*; w⁻)         trong mọi trường hợp còn lại, kể cả truncated
```

Ván bị cắt vì đạt `score_limit` (truncated) thì chim chưa chết, nên Target ở bước đó vẫn cộng giá trị tương lai.

Mã giả, với w là trọng số mạng chính, w⁻ là trọng số Target Network và D là Experience Replay:

```
Khởi tạo Q(·; w) ngẫu nhiên; w⁻ ← w; D rỗng
s ← env.reset()
lặp với frame = 1, 2, …, tổng số khung hình:
    ε ← lịch giảm tuyến tính
    a ← hành động ngẫu nhiên với xác suất ε, ngược lại argmax_a Q(s, a; w)
    s', r, terminated, truncated ← env.step(a)
    lưu (s, a, r, s', terminated) vào D
    nếu terminated hoặc truncated: s ← env.reset()
    ngược lại: s ← s'

    nếu |D| ≥ 10.000 và frame chia hết cho 4:
        lấy ngẫu nhiên 64 transition (sᵢ, aᵢ, rᵢ, s'ᵢ, terminatedᵢ) từ D
        aᵢ* ← argmax_a' Q(s'ᵢ, a'; w)
        Targetᵢ ← rᵢ nếu terminatedᵢ, ngược lại rᵢ + γ · Q(s'ᵢ, aᵢ*; w⁻)
        L ← trung bình theo i của (Targetᵢ − Q(sᵢ, aᵢ; w))²
        cập nhật w một bước theo gradient của L
        sau mỗi 1.000 lần cập nhật: w⁻ ← w

    cứ mỗi 25.000 khung hình: đánh giá với ε = 0 trên một môi trường riêng, lưu w nếu tốt nhất
```

Cách đánh giá:

- Mỗi lần đánh giá chạy 20 ván với ε = 0 trên một môi trường riêng, để không cắt ngang ván huấn luyện đang chạy; lần cuối chạy 100 ván. Dùng seed (số khởi tạo bộ sinh ngẫu nhiên, quyết định chuỗi khe) cố định và khác seed huấn luyện, để các lần đánh giá so sánh được với nhau. Cài đặt hiện tại chỉ đặt seed ở ván đầu; chuỗi khe của các ván sau phụ thuộc số ống đã sinh ở các ván trước, tức phụ thuộc chính sách đang được chấm. Muốn mọi checkpoint gặp cùng chuỗi khe thì phải đặt seed riêng cho từng ván.
- Đặt `score_limit` để agent giỏi không chơi mãi. Theo bảng tốc độ ở cuối mục này, một chu kỳ huấn luyện 25.000 khung hình mất khoảng 17 giây (6.250 lần cập nhật và 25.000 lần chọn hành động). Khi agent đã giỏi, 20 ván với giới hạn 200 dài khoảng 152.000 khung hình, tức khoảng 44 giây, gấp 2,6 lần chu kỳ huấn luyện; với giới hạn 1.000 là khoảng 760.000 khung hình, tức khoảng 220 giây, gấp 13 lần. Đo trên Colab: một chu kỳ 25.000 khung hình kèm đánh giá mất khoảng 1 phút khi agent còn kém và 3,1–4,3 phút khi cả 20 ván đạt 200 ống, nên đánh giá tốn khoảng 2–3 lần huấn luyện. Các lần đánh giá định kỳ nên dùng 200, lần cuối có thể dùng 1.000.
- Báo cáo trung bình, trung vị, thấp nhất và cao nhất của điểm, vì điểm dao động mạnh: luật viết tay đạt từ 4 đến 160 ống.
- Khi agent đã chơi tốt, Q của các trạng thái an toàn nên tiến về khoảng 11,9 đến 12,9, còn trạng thái sát va chạm có Q thấp hơn, xuống tới khoảng −1. Không trạng thái nào có giá trị thật cao hơn 12,9, nên Q vượt rõ mức này là dấu hiệu đánh giá quá cao hoặc việc học đang phân kỳ.
- Lưu checkpoint (bản lưu trọng số) tốt nhất theo điểm đánh giá, vì đường học của DQN thường không tăng đều. Điểm bị chặn ở `score_limit`, nên tiêu chí "lớn hơn hẳn mức tốt nhất cũ" giữ mãi lần đầu tiên chạm trần: trong lần chạy 10 triệu khung hình, checkpoint được giữ là ở 850.000. Khi điểm đã chạm trần cần tiêu chí khác, ví dụ đánh giá thêm ván hoặc với giới hạn cao hơn.

Về tốc độ, CPU là đủ. TensorFlow từ bản 2.11 không dùng GPU trên Windows, nên mọi phép tính chạy trên CPU. Đo trên máy phát triển trong env `flappybird`, 1 luồng và mạng như trên (69.891 tham số):

| Cách gọi mạng | Tốc độ đo được |
|---|---|
| Một bước cập nhật Double DQN, batch 64, bọc trong `tf.function` | khoảng 640 lần/giây |
| Chọn hành động cho 1 observation, bọc trong `tf.function` | khoảng 3.450 lần/giây |
| Chọn hành động bằng `model(x)` không bọc | khoảng 380 lần/giây |
| Chọn hành động bằng `model.predict(x)` | khoảng 22 lần/giây |

1 triệu khung hình ứng với 250.000 lần cập nhật (khoảng 6,5 phút) và tối đa 1 triệu lần chọn hành động (khoảng 5 phút), tức khoảng 11 đến 12 phút tính toán chưa kể thời gian chạy trò chơi. Đo trên Colab, 1 triệu khung hình đầu mất 70–82 phút, vì con số trên chưa tính chi phí chạy trò chơi và Python ở mỗi bước, cũng như đánh giá định kỳ. Vì vậy:

- Bọc hàm cập nhật và hàm chọn hành động trong `tf.function`. Tự viết bước cập nhật bằng `tf.GradientTape` thay vì `model.fit`, vì Target được tính từ hai mạng ở mỗi batch.
- Không dùng `model.predict` để chọn hành động từng khung hình: hàm này dành cho dữ liệu lớn và tốn chi phí khởi động ở mỗi lần gọi, nên 1 triệu khung hình sẽ mất khoảng 13 giờ.
- Đồng bộ Target Network bằng `target.set_weights(model.get_weights())`.
