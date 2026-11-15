# Định nghĩa bài toán FlappyBird và lựa chọn lý thuyết

FlappyBird là trò chơi một nút bấm. Con chim liên tục rơi xuống, và mỗi lần người chơi bấm, chim vỗ cánh bật lên. Các cặp ống chạy từ phải sang trái; mỗi cặp gồm một ống trên và một ống dưới, ở giữa có một khe trống. Chim phải bay lọt qua khe. Chạm ống hoặc chạm đất là thua, và điểm là số cặp ống đã vượt qua. Trong tài liệu này, "ống" luôn chỉ một cặp ống.

Đồ án xây dựng một chương trình AI, gọi là agent, tự học chơi trò này. Không ai dạy agent cách chơi: nó chơi thử rất nhiều ván, nhận điểm thưởng hoặc phạt sau mỗi hành động, và dần tìm ra cách chơi tốt. Cách học này gọi là học tăng cường (Reinforcement Learning, RL).

Phương pháp được chọn là DQN (Deep Q-Network) kèm hai cải tiến Double DQN và Dueling Network. Agent không biết luật chơi, còn số tình huống thì quá nhiều để ghi vào một bảng, nên agent dùng một mạng nơ-ron để học cách chấm điểm từng hành động. DQN là cách huấn luyện mạng đó một cách ổn định. Mục 3 sẽ đi qua 9 bước lập luận để đến lựa chọn này.

Môi trường là thư viện [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium) phiên bản 0.4.0. Lý thuyết lấy từ các tài liệu trong folder [Knowledge](Knowledge/). Số liệu đo được, công thức chi tiết và cấu hình cài đặt nằm ở [Implementation_Notes.md](Implementation_Notes.md).

## Thuật ngữ

| Từ | Nghĩa |
|---|---|
| Agent | Chương trình ra quyết định, tức "người chơi" AI |
| Môi trường | Trò chơi mà agent tương tác: nhận hành động, trả về tình huống mới và reward |
| Trạng thái | Tình huống hiện tại của trò chơi, phần agent nhìn thấy, gọi là observation |
| Hành động | Vỗ cánh hoặc không |
| Reward | Điểm thưởng (dương) hoặc phạt (âm) mà môi trường trả về sau mỗi bước |
| Episode | Một ván chơi, từ lúc bắt đầu đến khi chim va chạm |
| Transition | Ghi chép của một bước: trạng thái s, hành động a, reward r và trạng thái kế tiếp s' |
| Policy | Cách chơi, tức quy tắc chọn hành động ở mỗi trạng thái |
| Q(s, a) | Điểm của hành động a ở trạng thái s: tổng reward mong đợi nếu chọn a rồi chơi tiếp |

## 1. Bài toán

Mục tiêu là vượt càng nhiều ống càng tốt. Khi học, agent cố làm tổng reward lớn nhất (mục 2.3); khi chấm điểm, ta đếm số ống agent vượt được.

### 1.1. Agent nhìn thấy gì

Ở mỗi khung hình, agent nhận 12 con số. Chín số mô tả ba ống gần nhất, mỗi ống ba số: vị trí ngang, độ cao mép trên và độ cao mép dưới của khe. Ba số còn lại mô tả chim: độ cao, vận tốc theo chiều dọc và góc nghiêng. Chim đứng yên theo chiều ngang, chỉ có ống di chuyển.

Ba ống được xếp theo vị trí từ trái sang phải. Vì vậy ống chim vừa bay qua vẫn đứng đầu danh sách thêm một lúc, và trong lúc đó ống chim cần vượt lại nằm ở vị trí thứ hai. Agent phải tự nhận ra nên nhìn vào ống nào (Bước 6). Ống chưa vào màn hình thì bị che, còn ống chim cần vượt luôn hiển thị.

### 1.2. Agent làm được gì

Agent có hai hành động. IDLE là không làm gì, và trọng lực kéo chim rơi nhanh dần. FLAP là vỗ cánh: chim bật lên một đoạn rồi rơi xuống lại.

### 1.3. Trò chơi diễn biến thế nào

Biết tình huống hiện tại và hành động thì biết chính xác khung hình sau, trừ một điều: khe của mỗi ống mới được đặt ngẫu nhiên ở một trong 8 độ cao, với khả năng như nhau. Hai ống liên tiếp cách nhau khoảng 38 khung hình.

### 1.4. Thưởng và phạt

| Tình huống | Reward |
|---|---|
| Còn sống sau một khung hình | +0,1 |
| Vượt qua một ống | +1 |
| Đang ở phía trên mép màn hình | −0,5 |
| Va chạm (chạm ống hoặc chạm đất) | −1, và ván kết thúc |

### 1.5. Ván chơi kết thúc khi nào

Ván chỉ kết thúc khi chim va chạm, và không có giới hạn thời gian. Bay lên trên mép màn hình chỉ bị phạt; chim cũng không thể bay vòng qua phía trên ống, vì ống trên kéo dài lên cao hơn mức chim bay tới được.

Một agent đủ giỏi có thể chơi mãi. Tính toán trên luật chơi chép từ mã nguồn cho thấy dù các khe được xếp thế nào, luôn có cách vỗ cánh để không bao giờ va chạm, và cách đó chỉ cần những gì agent nhìn thấy. Phép tính này chỉ dùng để phân tích bài toán; bản thân agent không được biết luật chơi (mục 2.1).

### 1.6. Mốc so sánh

Để biết agent chơi tốt hay dở, cần hai cách chơi đơn giản làm mốc.

**Chơi ngẫu nhiên.** Ở mỗi khung hình, tung đồng xu: vỗ cánh hoặc không, mỗi bên một nửa. Cả 1.000 ván thử đều chết ở ống đầu tiên, với 0 điểm: chim vỗ cánh quá nhiều nên bay quá cao và đâm vào ống trên.

**Luật viết tay.** Đây là một quy tắc do người viết sẵn, không học gì cả. Ý tưởng giống cách người mới chơi: cứ giữ chim ở sát phía dưới khe của ống sắp tới, sắp rơi xuống thấp quá thì vỗ cánh. Cụ thể, ở mỗi khung hình luật làm ba việc:

1. Tìm ống cần vượt: ống gần nhất mà chim chưa bay qua hết.
2. Kẻ một vạch ngang cao hơn mép dưới khe của ống đó một khoảng nhỏ (12 pixel, tức khoảng nửa thân chim).
3. Đoán vị trí chim ở khung hình sau nếu không vỗ cánh. Nếu khi đó đáy chim sẽ thấp hơn vạch thì vỗ cánh, ngược lại thì không làm gì.

Luật chỉ dùng những gì agent cũng nhìn thấy (12 con số của observation, mục 1.1). Riêng bước 3 cần biết trọng lực kéo chim xuống thế nào, tức một phần luật chơi mà agent không được biết. Luật đạt trung bình 34,2 ống mỗi ván trên 200 ván, nhưng điểm dao động mạnh: có ván chỉ qua 4 ống, có ván qua hơn 150 ống.

Hai con số 0 và 34,2 là mốc để so sánh agent: agent học được thì phải vượt xa 0, và chơi tốt thì phải vượt 34,2. Với luật viết tay, chỉ 0,075% số bước là va chạm, nên trong dữ liệu va chạm là sự kiện rất hiếm.

Cách cài đặt chi tiết và số liệu đo được nằm ở [Implementation_Notes.md](Implementation_Notes.md) mục 4.

## 2. Mô hình MDP

### 2.1. Các thành phần

MDP (Markov Decision Process) là mô hình toán học chuẩn cho bài toán ra quyết định qua nhiều bước. Với FlappyBird:

| Thành phần | Ý nghĩa | Trong FlappyBird |
|---|---|---|
| S | Tập trạng thái | Các bộ 12 con số ở mục 1.1 |
| A | Tập hành động | {IDLE, FLAP} |
| T(s, a, s') | Xác suất sang trạng thái s' khi đang ở s và chọn a | Luật vật lý của trò chơi. Yếu tố ngẫu nhiên duy nhất là độ cao khe của ống mới (mỗi độ cao có xác suất 1/8), nên môi trường là stochastic |
| R(s, a, s') | Reward của bước chuyển đó | Bảng ở mục 1.4 |
| γ | Hệ số chiết khấu | 0,99 (mục 2.3) |
| Trạng thái kết thúc | Nơi ván dừng | Va chạm |

Agent không được biết T và R; nó chỉ thấy các transition (s, a, r, s') thu được khi chơi. T và R có trong mã nguồn trò chơi, nhưng dự án tách phần AI khỏi phần trò chơi, nên agent phải tự học như một người chơi thật.

### 2.2. Markov Property

Markov Property đòi hỏi trạng thái hiện tại chứa đủ thông tin để dự đoán tương lai mà không cần nhớ quá khứ. Muốn biết khung hình sau, chỉ cần độ cao và vận tốc của chim cùng vị trí và khe của các ống, và 12 con số chứa đủ những thứ đó. Agent vì vậy dùng thẳng observation làm trạng thái. Nó không cần ghép nhiều khung hình liên tiếp, vì vận tốc đã có sẵn, và không cần biết điểm số, vì luật chơi không phụ thuộc vào điểm. Ống bị che gần như không ảnh hưởng: khe của nó hoàn toàn ngẫu nhiên, còn ống cần vượt luôn hiển thị. Ngoại lệ nhỏ là vị trí ngang của ống bị che cũng không được cho biết, nên agent không biết chính xác ống đó sẽ xuất hiện sớm hay muộn vài khung hình.

### 2.3. Living Reward và Discounting

Reward +0,1 cho mỗi khung hình còn sống (Living Reward) khuyến khích agent sống lâu. Ống đến đều đặn và không né được, nên sống lâu cũng là vượt nhiều ống, và agent không có cách nào kiếm reward mà bỏ qua mục tiêu.

Vì có thể chơi mãi (mục 1.5), một ván có thể dài vô hạn (Infinite Horizon), kéo theo tổng reward cũng có thể vô hạn, và hai cách chơi cùng cho tổng vô hạn thì không so sánh được. Discounting giải quyết việc này: reward nhận sau k bước được nhân với γ^k, với 0 < γ < 1, nên phần thưởng càng xa càng nhẹ và tổng luôn hữu hạn. Hai cách khác không hợp với bài toán. Finite Horizon (giới hạn số bước) làm cách chơi tối ưu phụ thuộc vào số bước còn lại. Absorbing State (coi va chạm là điểm dừng hẳn) chỉ giữ tổng hữu hạn khi mọi cách chơi đều sớm muộn va chạm, mà ở đây có cách chơi không bao giờ va chạm.

Tài liệu chọn γ = 0,99. Hai ống cách nhau khoảng 38 khung hình, và với γ = 0,99, reward của ống kế tiếp còn giữ khoảng 2/3 trọng số (0,99^38 ≈ 0,68), nên agent vẫn "nhìn thấy" ống kế tiếp. Với γ = 0,9, trọng số đó chỉ còn 2%. Với γ = 0,999, agent nhìn xa gấp 10 lần mức cần thiết, và các giá trị phải học cũng lớn gấp 10 lần.

Với γ = 0,99, một agent không bao giờ chết nhận tổng reward có chiết khấu khoảng 12. Va chạm chỉ cho −1, nhưng nó còn làm mất khoảng 12 điểm tương lai, và phần này chiếm gần hết cái giá của việc chết. Mục 4 dựa vào điều này khi xử lý va chạm.

### 2.4. Value, Q-Value và Bellman Optimality Equation

V\*(s) là tổng reward có chiết khấu mong đợi khi bắt đầu ở trạng thái s rồi chơi tối ưu. Q\*(s, a) cũng vậy, nhưng bước đầu tiên bắt buộc chọn hành động a. Bellman Optimality Equation nối hai bước liên tiếp: giá trị của việc chọn a ở s bằng reward ngay lập tức cộng γ nhân giá trị tốt nhất ở trạng thái kế tiếp, lấy trung bình theo các khả năng ngẫu nhiên. Nếu trạng thái kế tiếp là va chạm thì phần sau bằng 0.

```
Q*(s,a) = E[ r + γ · max_a' Q*(s',a') ]
π*(s)   = argmax_a Q*(s,a)
```

E[·] là trung bình mong đợi, còn π\*(s) là cách chơi tối ưu: chọn hành động có Q\* lớn nhất. Nếu biết Q\*, agent chỉ cần so sánh Q\*(s, FLAP) với Q\*(s, IDLE) ở mỗi khung hình. Mục 3 chọn cách ước lượng Q\* khi không biết T và R.

## 3. Chọn phương pháp qua 9 bước

```
Bài toán MDP (mục 2)
 │  Bước 1: agent không biết T, R; bài toán quá lớn → không giải thẳng
 ▼
Học từ kinh nghiệm chơi (Online Learning)
 │  Bước 2: luật chơi tự học sẽ sai dần → model-free
 │  Bước 3: chưa có cách chơi tốt để đánh giá → active
 │  Bước 4: không có T để nhìn trước → học Q thay vì V
 ▼
Học Q(s, a)
 │  Bước 5: cách chơi cuối không được ngẫu nhiên; cần dùng lại dữ liệu cũ → Q-Learning
 ▼
Q-Learning
 │  Bước 6: bảng quá lớn, hàm tuyến tính không đủ → mạng nơ-ron (Q-Network)
 ▼
Q-Learning + Q-Network
 │  Bước 7: dữ liệu liên tiếp giống nhau; mục tiêu tự dịch chuyển
 ▼
DQN = Q-Network + Experience Replay + Target Network
 │  Bước 8: hai hành động có giá trị sát nhau; va chạm hiếm
 ▼
DQN + Double DQN + Dueling Network

Bước 9: so với các hướng khác, không hướng nào hợp hơn.
```

### Bước 1. Có giải thẳng bài toán được không?

Value Iteration và Policy Iteration (gồm cả Policy Evaluation) là quy hoạch động: chúng áp dụng lặp lại phương trình Bellman lên mọi trạng thái cho tới khi giá trị không đổi. Expectimax thì tìm kiếm trên cây các diễn biến có thể xảy ra. Các phương pháp này đều cần biết T và R, mà agent không biết (mục 2.1). Kể cả khi biết, bài toán có cỡ 100 triệu trạng thái nếu tính đến từng pixel, nên phải giữ một bảng cỡ đó và quét lại hàng trăm lần.

Agent vì vậy phải **học từ kinh nghiệm chơi** (Online Learning).

### Bước 2. Học lại luật chơi, hay học thẳng cách chơi?

Model-based học lại T và R từ kinh nghiệm rồi lập kế hoạch trên luật vừa học. Model-free bỏ qua luật chơi và học thẳng V hoặc Q.

Ở FlappyBird, hậu quả của một quyết định thường chỉ lộ ra khi chim tới ống kế tiếp, khoảng 38 khung hình sau. Luật tự học phải dự đoán đúng chừng ấy bước liên tiếp, trong khi sai số nhỏ ở mỗi bước cộng dồn lại (Compounding Errors) và va chạm được xét đến từng pixel. Có luật rồi, agent vẫn phải lập kế hoạch lại ở mỗi khung hình, tốn hơn nhiều so với việc tính Q một lần.

Hướng được chọn là **model-free**.

### Bước 3. Đánh giá một cách chơi có sẵn, hay tự tìm cách chơi?

Passive RL chỉ đánh giá một policy cho trước. Active RL tự chọn hành động để tìm policy tốt nhất, và phải cân bằng giữa thử hành động mới để biết thêm (exploration) với dùng hành động đang tốt nhất (exploitation). FlappyBird chưa có cách chơi tốt nào để đánh giá; tìm ra nó chính là mục tiêu.

Agent cần **Active RL**. Passive RL chỉ còn dùng để chấm điểm: cho agent chơi nhiều ván rồi lấy điểm trung bình (Direct Evaluation, mục 4).

### Bước 4. Học V(s) hay Q(s, a)?

Muốn chọn hành động từ V(s), agent phải biết mỗi hành động dẫn tới trạng thái nào, tức cần T. Chọn từ Q(s, a) chỉ cần so sánh các giá trị, và agent model-free không có T (Bước 2).

Agent **học Q(s, a)** rồi chọn hành động có Q lớn nhất (Policy Extraction). Với hai hành động, đó chỉ là so sánh hai số.

### Bước 5. Q-Learning hay SARSA?

Cả hai thuật toán thuộc họ TD Learning. Sau mỗi transition, chúng kéo Q(s, a) lại gần một giá trị mục tiêu gọi là Target:

```
Q(s,a) ← Q(s,a) + α · (Target − Q(s,a))       α: learning rate
```

Target bằng reward vừa nhận cộng γ nhân ước lượng Q ở trạng thái kế tiếp, hoặc chỉ bằng reward nếu trạng thái kế tiếp là va chạm. Việc dùng chính ước lượng của mình để tạo mục tiêu gọi là bootstrapping, còn hiệu Target − Q gọi là TD Error.

Hai thuật toán khác nhau ở Target. Q-Learning dùng r + γ · max_a' Q(s', a'), tức giả định từ bước sau agent chơi tốt nhất. Nhờ vậy nó học được cách chơi tốt nhất kể cả khi dữ liệu đến từ một cách chơi khác, và được gọi là off-policy. SARSA dùng r + γ · Q(s', a') với a' là hành động agent thực sự chọn, nên nó học giá trị của chính cách chơi đang dùng (on-policy).

Khi học, agent phải thử ngẫu nhiên. Cách phổ biến là ε-greedy: với xác suất nhỏ ε thì chọn ngẫu nhiên, còn lại chọn hành động có Q lớn nhất. Khi chấm điểm, agent thôi thử ngẫu nhiên, nên thứ cần học là giá trị của cách chơi không ngẫu nhiên. Q-Learning học đúng giá trị đó (Q\*), còn SARSA học giá trị của cách chơi vẫn còn ngẫu nhiên. Tính off-policy còn cho phép dùng lại dữ liệu cũ do các phiên bản trước của agent tạo ra, và đây là điều kiện để dùng Experience Replay ở Bước 7. SARSA có lợi khi agent còn thử ngẫu nhiên nhiều, vì nó học cách chơi thận trọng hơn, nhưng lợi thế này mất dần khi ε giảm về gần 0.

Thuật toán được chọn là **Q-Learning**.

### Bước 6. Lưu Q bằng cách nào?

Có ba cách lưu Q. Q-Table là một bảng có một ô cho mỗi cặp (trạng thái, hành động). Approximate Q-Learning tính Q bằng tổng có trọng số của các đặc trưng (feature), tức các con số mô tả trạng thái. Q-Network tính Q bằng mạng nơ-ron.

Bảng không dùng được: có cỡ 100 triệu trạng thái, quá nhiều để ghé hết, và bảng không tự suy ra giá trị cho trạng thái chưa gặp (Generalization). Tổng có trọng số trên 12 con số thô cũng không đủ, vì có hai việc nó không làm được. Việc thứ nhất là chọn xem ống cần vượt ở vị trí thứ nhất hay thứ hai (mục 1.1), một phép "nếu… thì…" mà phép cộng có trọng số không biểu diễn được. Việc thứ hai là tạo ra một Q chỉ cao khi chim ở trong một dải quanh khe và thấp ở cả hai phía, trong khi tổng có trọng số chỉ tăng dần hoặc giảm dần theo độ cao. Muốn dùng cách này phải tự thiết kế đặc trưng.

Mạng nơ-ron có các lớp phi tuyến nên tự học được cả hai việc từ 12 con số thô (Representation Learning). Mạng được cập nhật như một bản mở rộng của Approximate Q-Learning: Gradient Descent giảm bình phương sai số (Target − Q)² (Least Squares), với gradient tính bằng Backpropagation.

Q được lưu bằng **Q-Network** dạng MLP, tức các lớp nơ-ron nối đầy đủ với nhau. CNN (mạng tích chập) dùng chung trọng số giữa các vị trí (Weight Sharing), có ích khi cùng một mẫu lặp lại ở nhiều chỗ như trong ảnh, còn ở đây đầu vào chỉ có 12 con số. RNN (mạng có bộ nhớ qua các bước) cũng không cần, vì 12 con số đã đủ thông tin (mục 2.2).

### Bước 7. Làm sao để việc học ổn định?

Ghép Q-Learning với mạng nơ-ron có thể làm việc học dao động hoặc hỏng hẳn. Có hai nguyên nhân, và cả hai đều xuất hiện trong FlappyBird.

| Vấn đề | Trong FlappyBird | Cách giải |
|---|---|---|
| Dữ liệu liên tiếp gần như giống nhau (Correlations Between Samples) | Hai khung hình liền nhau chỉ khác vài pixel. Học theo đúng thứ tự, mạng sẽ khớp đoạn đang chơi và quên các đoạn trước | Experience Replay: lưu nhiều transition gần đây vào bộ nhớ, mỗi lần học lấy ngẫu nhiên một nhóm (batch) trong đó. Cách này dùng được vì Q-Learning là off-policy |
| Mục tiêu dịch chuyển theo chính mạng đang học (Non-Stationary Targets) | s và s' gần giống nhau. Khi mạng tăng Q(s, a), Q(s', ·) cũng tăng gần bằng chừng ấy và kéo Target lên theo, nên sai số gần như không giảm | Target Network: tính Target bằng một bản sao của mạng, và chỉ cập nhật bản sao sau một số lần học |

Ghép ba phần lại được **DQN**: Q-Network, Experience Replay và Target Network.

### Bước 8. Thêm cải tiến nào?

Ở đa số khung hình, vỗ cánh hay không gần như tốt ngang nhau, vì sai một khung hình thường sửa được ngay sau đó. Khi hai giá trị gần bằng nhau, lấy số lớn hơn trong hai ước lượng có sai số sẽ hay chọn trúng ước lượng bị lệch lên. Q vì thế bị đánh giá quá cao, và sai lệch này còn cộng dồn qua bootstrapping. Double DQN sửa việc này bằng cách để mạng chính chọn hành động tốt nhất ở s', còn Target Network chấm điểm hành động đó. Hai mạng có sai số khác nhau nên ít khi cùng lệch lên.

Q(s, a) gồm một phần chung lớn cho cả hai hành động và một phần chênh lệch nhỏ giữa chúng. Dueling Network chia đầu ra của mạng thành V(s) cho phần chung và A(s, a) cho phần chênh lệch (advantage). Nhờ vậy phần chung được học từ mọi transition, bất kể agent đã chọn hành động nào.

Va chạm rất hiếm trong dữ liệu (mục 1.6). Prioritised Replay lấy thường xuyên hơn những mẫu mà mạng đoán sai nhiều, nên gặp mẫu va chạm nhiều hơn, đổi lại cài đặt phức tạp hơn. Nó chỉ nên được thêm khi điểm đánh giá tăng rồi tụt và không hồi phục, dấu hiệu agent quên cách tránh va chạm. Một lần tụt rồi hồi phục ngay ở lần đánh giá sau chưa đủ, vì nó có thể chỉ là nhiễu của việc chấm trên ít ván.

Phương pháp cuối cùng là **DQN kèm Double DQN và Dueling Network**.

### Bước 9. Kiểm tra ngược: hướng khác có hợp hơn không?

| Hướng | Vì sao không chọn |
|---|---|
| Policy Gradient, Actor-Critic, A3C: học thẳng xác suất chọn từng hành động | Các phương pháp này là on-policy nên không dùng lại được dữ liệu cũ, trong khi mẫu va chạm vốn đã hiếm. Với hai hành động, học Q rồi so sánh hai số đơn giản hơn |
| Deep DPG (Deterministic Policy Gradient) | Dành cho hành động liên tục, như góc lái; FlappyBird chỉ có hai hành động |
| Fictitious Self-Play | Dành cho trò chơi nhiều người chơi; FlappyBird chỉ có một |
| Model-Based Deep RL | Gặp vấn đề sai số cộng dồn ở Bước 2. Còn nếu dùng luật chép từ mã nguồn thì agent chỉ lập kế hoạch, trái với mục tiêu tự học của đồ án |
| Huấn luyện song song, bất đồng bộ (Gorila, Asynchronous RL) | Các phương pháp này chạy nhiều agent cùng lúc để lấy dữ liệu nhanh hơn và bớt tương quan. Ở đây trò chơi đã chạy rất nhanh, phần chậm là cập nhật mạng, và Experience Replay đã xử lý tương quan |
| CNN trên ảnh màn hình | 12 con số đã đủ thông tin; học từ ảnh chỉ làm bài toán khó hơn |

### Tổng hợp

- Dùng trực tiếp: MDP; Markov Property; Living Reward, Discounting, Infinite Horizon; V, Q, Bellman Optimality Equation; Policy Extraction; Online Learning; Model-Free, Active RL; Exploration, Exploitation, ε-greedy; Q-Learning, off-policy; mạng nơ-ron, Representation Learning, Gradient Descent, SGD, Backpropagation, Overfitting; Q-Network, DQN, Experience Replay, Target Network; Double DQN; Dueling Network.
- Làm nền tảng: TD Learning; Generalization, Feature, Approximate Q-Learning; Least Squares.
- Chỉ để chấm điểm: Passive RL, Direct Evaluation.
- Tùy chọn: Prioritised Replay.
- Không dùng: Policy Evaluation, Value Iteration, Policy Iteration, Expectimax (Bước 1); Model-Based Learning (Bước 2); SARSA (Bước 5); CNN, RNN, Weight Sharing (Bước 6); Policy Gradient, Actor-Critic, A3C, Deep DPG, Fictitious Self-Play, Model-Based Deep RL, huấn luyện phân tán (Bước 9).

## 4. Phương pháp được chọn và cách đánh giá

Agent DQN cho FlappyBird gồm bốn phần:

- Q-Network là một MLP nhận 12 con số và trả về hai giá trị Q(s, IDLE) và Q(s, FLAP), với đầu ra chia theo kiểu Dueling. Mạng được huấn luyện bằng Adam, một biến thể của Stochastic Gradient Descent (SGD), và giữ ở kích thước vừa phải để tránh Overfitting, tức học thuộc lòng dữ liệu đang có mà kém ở tình huống mới.
- Agent chọn hành động bằng ε-greedy. ε bắt đầu ở 1, tức hoàn toàn ngẫu nhiên, rồi giảm dần về một số rất nhỏ. Mỗi ống cần khoảng 38 quyết định liên tiếp, nên khi ε còn lớn thì gần như ống nào cũng có vài quyết định ngẫu nhiên.
- Mỗi transition được lưu vào Experience Replay. Định kỳ, agent lấy ngẫu nhiên một batch, tính Target theo Double DQN rồi cập nhật mạng để giảm (Target − Q)².
- Target Network được chép lại từ mạng chính sau một số lần cập nhật.

Khi chim va chạm, Target chỉ là reward −1. Nếu lỡ cộng thêm giá trị tương lai như ở các bước khác, Target của bước va chạm sẽ vào khoảng 11, gần bằng giá trị sống tiếp (khoảng 12, mục 2.3), và agent gần như không thấy chết là xấu.

Lúc đầu agent chơi gần như ngẫu nhiên và không bao giờ ghi điểm (mục 1.6), nhưng vẫn học được. Mức phạt −0,5 khi bay lên trên mép màn hình cho tín hiệu ngay từ đầu, vì vỗ cánh quá nhiều thì bị phạt. Khi agent bớt vỗ cánh, các ván bắt đầu dài ngắn khác nhau, và +0,1 mỗi khung hình thưởng cho ván dài hơn. Agent học tránh bay quá cao và tránh rơi, sống lâu hơn, rồi mới gặp reward +1. Nếu khi thử ngẫu nhiên agent vỗ cánh ít hơn một nửa số lần, một phần nhỏ các ván ngẫu nhiên đã vượt được ống đầu tiên, và agent gặp reward +1 sớm hơn ([Implementation_Notes.md](Implementation_Notes.md) mục 5).

Để chấm điểm, agent chơi nhiều ván không ngẫu nhiên (ε = 0), rồi ta lấy số ống trung bình mỗi ván. Mốc so sánh là 0 cho chơi ngẫu nhiên và 34,2 cho luật viết tay (mục 1.6).

Kích thước mạng, các siêu tham số, mã giả và cách đánh giá chi tiết nằm ở [Implementation_Notes.md](Implementation_Notes.md).

<!-- ## 5. Hạn chế

- Cấu hình trong [Implementation_Notes.md](Implementation_Notes.md) được chọn bằng lập luận. Nó đã được chạy 3 lần và agent học được, vượt xa luật viết tay ([README.md](README.md) mục 6), nhưng chưa được so sánh với cấu hình nào khác. Cũng chưa có lần chạy bỏ Double DQN hoặc Dueling Network, nên lập luận ở Bước 8 chưa có bằng chứng thực nghiệm riêng.
- Ba lần chạy cùng cấu hình cho kết quả chênh nhau nhiều ở cùng thời điểm huấn luyện, nên một lần chạy không đủ để kết luận về một cấu hình.
- Các tỷ lệ đo được, như tần suất va chạm, lấy từ luật viết tay; dữ liệu của DQN sẽ khác.
- Q-Learning chỉ được bảo đảm hội tụ khi dùng bảng. Với mạng nơ-ron thì không có bảo đảm đó, và Experience Replay cùng Target Network chỉ làm giảm sự mất ổn định.
- Kết luận "có thể chơi mãi" được tính trên luật chơi của phiên bản 0.4.0. Nó cho biết bài toán có lời giải hoàn hảo; DQN có học được lời giải đó hay không thì vẫn phải thử. -->
