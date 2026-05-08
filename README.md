# Phát hiện mã độc Android sử dụng Học sâu với Dữ liệu Dương tính và Chưa gán nhãn (PU Learning)



Dự án này tập trung vào việc ứng dụng mô hình Deep Generative Positive-Unlabeled Learning (VAE-PU) để phát hiện mã độc trên hệ điều hành Android. Nghiên cứu tập trung giải quyết bài toán trong điều kiện thực tế khắc nghiệt: giả định NON-SCAR (Non-Selected Completely At Random). Trong kịch bản này, xác suất một mẫu mã độc được gán nhãn (Positive) không cố định mà phụ thuộc trực tiếp vào các đặc trưng nội tại của nó. Những mẫu mã độc tinh vi, có hành vi che giấu thường có xác suất được gán nhãn thấp hơn, dẫn đến sự mất cân bằng và sai lệch nghiêm trọng trong dữ liệu huấn luyện. Bằng cách sử dụng mô hình sinh (Generative Model), dự án hướng tới việc học phân phối cốt lõi của dữ liệu để vượt qua rào cản về sự sai lệch gán nhãn này.



##  Tổng quan đề tài

* **Tên đề tài:** Ứng dụng mô hình học sâu trong phát hiện mã độc trên hệ điều hành Android bằng dữ liệu dương tính và dữ liệu chưa gán nhãn.

* **Mô hình cốt lõi:** VAE-PU (Variational Autoencoder for Positive-Unlabeled Learning).

* **Mục tiêu:** Nâng cao khả năng tổng quát hóa của mô hình đối với các mẫu mã độc tinh vi trong môi trường thiếu hụt dữ liệu lành tính và gán nhãn không ngẫu nhiên.



##  Môi trường thực nghiệm

Dự án được cấu hình và huấn luyện trên môi trường **Kaggle**:

* **Hardware:** GPU T4 x 2 (Dual-GPU).

* **Software:** Python 3.x, TensorFlow, Keras.

* **Dependencies:** Chi tiết tại `requirements.txt`.



##  Cấu trúc thư mục dự án

Việc tổ chức các file dựa trên mã nguồn gốc của tác giả Byeonghu-na và bổ sung các kịch bản thực nghiệm riêng:



```text

├── src/                    # Mã nguồn cốt lõi của mô hình

│   ├── model.py            # Định nghĩa kiến trúc mạng VAE-PU

│   ├── train.py            # Script huấn luyện mô hình chính

│   ├── main.py             # Điểm vào của ứng dụng

│   ├── config.py           # Cấu hình các tham số hệ thống

│   └── train_load.py       # Hỗ trợ vẽ biểu đồ phân phối xác suất của mạng mã hóa

├── experiments/            # Các file chạy kịch bản thực nghiệm

│   ├── 01_Model_Robustness_and_Sensitivity_Analysis.py   # Kiểm tra độ nhạy của mô hình

│   ├── 02_Temporal_Concept_Drift_2014_2025.py            # Kiểm tra mã độc theo thời gian

|   ├── 03_Seen_vs_Unseen_Family_Evolution.py             # Phân tích họ mã độc Đã biết (Seen) và Mẫu mới (Unseen)

└── requirements.txt        # Danh sách thư viện cần thiết

```



##  Hướng dẫn cài đặt và Thực nghiệm



Dự án này được thiết kế để chạy tối ưu trên môi trường **Kaggle Kernel** nhằm tận dụng sức mạnh của **Dual GPU T4**.



### 1. Cấu trúc mã nguồn trên Kaggle

Để đảm bảo tính module hóa, mã nguồn trong thư mục `src/` được tổ chức theo cấu trúc:

* **Modulized Code:** Mỗi tệp tin `.py` (như `model.py`, `train.py`) tương ứng với một đơn vị xử lý logic riêng biệt. 

* **Thực thi:** Bạn có thể import các module này vào Kaggle Notebook hoặc copy nội dung mỗi file vào các Code Cell riêng lẻ để bắt đầu huấn luyện(lưu ý mở comment các tại #%%writefile <tên file> để kaggle có thể lưu module vào RAM).



### 2. Huấn luyện và Lưu trữ mô hình (Training & Persistence)

* **Save Version:** Sử dụng tính năng **"Save Version"** (chọn chế độ *Save & Run All*) trên Kaggle để thực hiện huấn luyện toàn bộ mô hình.

* **Output:** Sau khi hoàn thành, các file trọng số mô hình (`.h5`, `.pth` hoặc checkpoints) sẽ được lưu tại thư mục `/kaggle/working/` hoặc thư mục `results/` như đã cấu hình.



### 3. Tái sử dụng mô hình cho các kịch bản thực nghiệm

Để sử dụng mô hình đã huấn luyện cho các kịch bản trong `/notebooks` (như kiểm tra mã độc theo thời gian):

1. Thêm kết quả đầu ra của Notebook huấn luyện vào Notebook thực nghiệm mới thông qua tính năng **"Add Data"** -> **"Notebook Output Files"**.

2. Nạp trọng số mô hình đã lưu vào Notebook thực nghiệm để tiến hành đánh giá mà không cần huấn luyện lại từ đầu.





##  Quy trình Đánh giá thực nghiệm 



Để đánh giá toàn diện khả năng tổng quát hóa của mô hình VAE-PU trên dữ liệu Android, dự án thực hiện kiểm tra thông qua 03 kịch bản dữ liệu khác nhau tại notebook `eval_test_scenarios.ipynb`.



### 1. Cấu trúc kịch bản kiểm tra 

Mô hình sau khi huấn luyện sẽ được đánh giá trên tập dữ liệu đầu vào:*  **[Link Google Drive chứa Dataset]**:https://drive.google.com/drive/folders/1DshTcKRxcTXX8vAlXrPUFadKcHRR90T1?usp=sharing



| Kịch bản | Thành phần dữ liệu | Mục tiêu đánh giá |
| :--- | :--- | :--- |
| **Kịch bản 1** | Toàn bộ các họ mã độc từ tập PL, PU và Test | Đánh giá độ chính xác tổng thể trên toàn bộ không gian mẫu. |
| **Kịch bản 2** | Chỉ bao gồm các họ mã độc thuộc tập PL và Test | Đánh giá khả năng nhận diện lại các họ đã biết kết hợp với các họ mới. |
| **Kịch bản 3** | Chỉ bao gồm các họ mã độc mới (chưa xuất hiện trong huấn luyện) | Kiểm tra khả năng phát hiện mã độc "Unseen" (chưa từng thấy). |



### 2. Phân tích khả năng phát hiện nâng cao

Sau khi có kết quả từ các kịch bản trên, mô hình tiến hành một bước phân tích chuyên sâu (Deep Analysis):

* **Đối tượng:** Các mẫu mã độc có chỉ số VirusTotal (`vt_count`) nằm trong khoảng `(4, 12]`.

* **Mục tiêu:** Kiểm chứng độ nhạy của mô hình đối với các mã độc "mới nổi" hoặc có hành vi che giấu tinh vi (thường có số lượng engine phát hiện thấp).

* **Đầu ra:** Biểu đồ phân tích tỷ lệ phát hiện (Detection Rate) ứng với từng mức độ nguy hiểm của mã độc.



### 3. Đánh giá sự tiến hóa của mã độc 



Để kiểm chứng khả năng phát hiện mã độc trong dài hạn, mô hình được thử nghiệm với dữ liệu xuyên suốt 11 năm (2014-2025, loại biên 2015).
tập dữ liệu đầu vào:*  


#### Kịch bản 4: Kiểm tra sự biến đổi khái niệm 

* **Dữ liệu đầu vào:** https://drive.google.com/drive/folders/1n4zk0hEg0pZEaMFKcaYdKXfQuobQZSMg?usp=sharing

* **Mục tiêu:** Đánh giá độ chính xác (Accuracy) và F1-Score của mô hình qua từng năm. Kết quả này giúp xác định thời điểm mô hình bắt đầu giảm phong độ do sự xuất hiện của các kỹ thuật gây nhiễu hoặc hành vi mã độc mới.



#### Kịch bản 5: Phân tích họ mã độc Đã biết (Seen) và Mẫu mới (Unseen)

Kịch bản này đi sâu vào việc phân loại khả năng nhận diện dựa trên lịch sử xuất hiện của các họ mã độc (Malware Families).

* **Dữ liệu đầu vào:** https://drive.google.com/drive/folders/1HmkYrEQGGmArXWaOGfNQhABtffqW7fo6?usp=sharing

* **Tiêu chí phân loại:**

    * **Seen Families:** Các họ mã độc đã từng xuất hiện trong tập dữ liệu năm 2013 (dữ liệu gốc/huấn luyện).

    * **Unseen Families:** Các họ mã độc hoàn toàn mới, xuất hiện lần đầu trong giai đoạn 2014-2025.

* **Mục tiêu:** Chứng minh sức mạnh của phương pháp **PU Learning** trong việc phát hiện các biến thể mã độc chưa từng có tiền lệ (Zero-day/New families) so với các mẫu đã biết.



##  Trích dẫn

Dự án có kế thừa và phát triển từ công trình:



> Na, Byeonghu, et al. "Deep Generative Positive-Unlabeled Learning under Selection Bias." CIKM '20.
