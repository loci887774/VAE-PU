# Phát hiện mã độc Android sử dụng Học sâu với Dữ liệu Dương tính và Chưa gán nhãn (PU Learning)



Dự án này tập trung vào việc ứng dụng mô hình Deep Generative Positive-Unlabeled Learning (VAE-PU) để phát hiện mã độc trên hệ điều hành Android. Nghiên cứu tập trung giải quyết bài toán trong điều kiện thực tế khắc nghiệt: giả định NON-SCAR (Non-Selected Completely At Random). Trong kịch bản này, xác suất một mẫu mã độc được gán nhãn (Positive) không cố định mà phụ thuộc trực tiếp vào các đặc trưng nội tại của nó. Những mẫu mã độc tinh vi, có hành vi che giấu thường có xác suất được gán nhãn thấp hơn, dẫn đến sự mất cân bằng và sai lệch nghiêm trọng trong dữ liệu huấn luyện. Bằng cách sử dụng mô hình sinh (Generative Model), dự án hướng tới việc học phân phối cốt lõi của dữ liệu để vượt qua rào cản về sự sai lệch gán nhãn này.



##  Tổng quan đề tài

* **Tên đề tài:** Ứng dụng mô hình học sâu trong phát hiện mã độc trên hệ điều hành Android bằng dữ liệu dương tính và dữ liệu chưa gán nhãn.

* **Mô hình cốt lõi:** VAE-PU (Variational Autoencoder for Positive-Unlabeled Learning).

* **Xử lý Bias:** Giải quyết vấn đề gán nhãn không ngẫu nhiên (Selection Bias) trong dữ liệu mã độc thực tế.

* **Mục tiêu:** Nâng cao khả năng tổng quát hóa của mô hình đối với các mẫu mã độc tinh vi trong môi trường thiếu hụt dữ liệu lành tính và gán nhãn không ngẫu nhiên.



##  Môi trường thực nghiệm

Dự án được cấu hình và huấn luyện trên môi trường **Kaggle**:

* **Phần cứng:** GPU T4 x 2 (Dual-GPU).

* **Framework:** TensorFlow, Keras.

* **Yêu cầu:** Chi tiết tại `requirements.txt`.


## Tập dữ liệu gốc
Tập dữ liệu gốc chưa qua xử lý được lưu trữ riêng biệt để đảm bảo tính toàn vẹn và phục vụ việc tái lập nghiên cứu: https://drive.google.com/drive/folders/1c0bQXwbKso6kCIwY-jetffSjz3z23m_b?usp=sharing.

##  Cấu trúc thư mục dự án

Dự án kế thừa kiến trúc từ Byeonghu-na và mở rộng các kịch bản thực nghiệm:



```text

├── src/                    # Mã nguồn cốt lõi của mô hình
│   ├── model.py            # Định nghĩa kiến trúc mạng VAE-PU
│   ├── train.py            # Script huấn luyện mô hình chính
│   ├── main.py             # Điểm vào của ứng dụng
│   ├── config.py           # Cấu hình các tham số hệ thống
│   └── train_load.py       # Hỗ trợ vẽ biểu đồ phân phối xác suất của mạng mã hóa
├── experiments/            # Các file chạy kịch bản thực nghiệm
│   ├── 01_Model_Robustness_and_Sensitivity_Analysis.py   # Kiểm tra độ nhạy của mô hình với các họ mã độc chưa biết và nhóm mã độc có vt_count trong khoảng 4-12
│   ├── 02_Temporal_Concept_Drift_2014_2025.py            # Kiểm tra khả năng phân loại mã độc của mô hình theo thời gian
|   ├── 03_Seen_vs_Unseen_Family_Evolution.py             # Kiểm tra khả năng phân loại họ mã độc Đã biết (Seen) và Mẫu mới (Unseen) theo thời gian
└── requirements.txt        # Danh sách thư viện cần thiết

```



##  Hướng dẫn cài đặt và Thực nghiệm



Dự án này được thiết kế để chạy tối ưu trên môi trường **Kaggle Kernel** nhằm tận dụng sức mạnh của **Dual GPU T4**.



### 1. Trên Kaggle

1. Upload thư mục src/ vào Kaggle Dataset cá nhân hoặc copy trực tiếp.

2. Sử dụng lệnh %%writefile để quản lý module trong Notebook.

3. Chế độ huấn luyện: Chọn Save & Run All để thực thi toàn trình trên Dual T4.

* **Tập dữ liệu huấn luyện:** https://drive.google.com/drive/folders/1iz2yLYONwFb6krfLGi5h3RcTU988f8za?usp=sharing



### 2. Đánh giá

Để tái sử dụng trọng số mô hình cho các tệp trong /experiments:

1. Dùng tính năng "Add Data" -> "Notebook Output Files" để liên kết file checkpoint của mô hình.

2. Sử dụng đúng các tập dữ liệu theo từng kịch bản đánh giá.


## Siêu tham số

Các siêu tham số này được lựa chọn thông qua quá trình huấn luyện mô hình và đánh giá để đưa ra lựa chọn.
* **Tập dữ liệu tìm các siêu tham số tối ưu:** https://drive.google.com/drive/folders/1u_0PKc3KqJpp_ZSRlz1uxGcENft6KIGY?usp=sharing


Siêu tham số |  SCAR  | NON-SCAR | 
--------- | --------- | --------- | 
alpha_gen | 0.3 | 1 | 
alpha_disc | 0.3 | 1 | 
alpha_gen2 | 1 | 2 | 



##  Quy trình Đánh giá thực nghiệm 



Để đánh giá toàn diện khả năng tổng quát hóa của mô hình VAE-PU trên dữ liệu, dự án thực hiện kiểm tra thông qua 03 kịch bản dữ liệu khác nhau.



### 1. Kiểm tra độ nhạy của mô hình với các họ mã độc chưa biết

* **Dữ liệu đầu vào:** https://drive.google.com/drive/folders/1DshTcKRxcTXX8vAlXrPUFadKcHRR90T1?usp=sharing



| Kịch bản | Thành phần dữ liệu | Mục tiêu đánh giá |
| :--- | :--- | :--- |
| **Kịch bản 1** | Toàn bộ các họ mã độc từ tập PL, PU và Test | Đánh giá độ chính xác tổng thể trên toàn bộ không gian mẫu. |
| **Kịch bản 2** | Chỉ bao gồm các họ mã độc thuộc tập PL và Test | Đánh giá khả năng nhận diện lại các họ đã biết kết hợp với các họ mới. |
| **Kịch bản 3** | Chỉ bao gồm các họ mã độc mới (chưa xuất hiện trong huấn luyện) | Kiểm tra khả năng phát hiện mã độc "Unseen" (chưa từng thấy). |



### 2. Phân tích khả năng phát hiện nâng cao

Sau khi có kết quả từ các kịch bản trên, mô hình tiến hành một bước phân tích chuyên sâu:

* **Đối tượng:** Các mẫu mã độc có chỉ số VirusTotal (`vt_count`) nằm trong khoảng `[4, 12]`.

* **Mục tiêu:** Kiểm chứng độ nhạy của mô hình đối với các mã độc "mới nổi" hoặc có hành vi che giấu tinh vi (thường có số lượng engine phát hiện thấp).




### 3. Đánh giá sự tiến hóa của mã độc 



Để kiểm chứng khả năng phát hiện mã độc trong dài hạn, mô hình được thử nghiệm với dữ liệu xuyên suốt 11 năm (2014-2025, bỏ 2015).
tập dữ liệu đầu vào:*  


#### Kịch bản 4: Kiểm tra sự biến đổi khái niệm 

* **Dữ liệu đầu vào:** https://drive.google.com/drive/folders/1n4zk0hEg0pZEaMFKcaYdKXfQuobQZSMg?usp=sharing

* **Mục tiêu:** Đánh giá độ chính xác (Accuracy) và F1-Score của mô hình qua từng năm. Kết quả này giúp xác định thời điểm mô hình bắt đầu giảm phong độ do sự xuất hiện của các kỹ thuật gây nhiễu hoặc hành vi mã độc mới.



#### Kịch bản 5: Phân tích họ mã độc Đã biết (Seen) và Mẫu mới (Unseen)

Kịch bản này đi sâu vào việc phân loại khả năng nhận diện dựa trên lịch sử xuất hiện của các họ mã độc.

* **Dữ liệu đầu vào:** https://drive.google.com/drive/folders/1HmkYrEQGGmArXWaOGfNQhABtffqW7fo6?usp=sharing

* **Tiêu chí phân loại:**

    * **Seen Families:** Các họ mã độc đã từng xuất hiện trong tập dữ liệu năm 2013 (dữ liệu gốc/huấn luyện).

    * **Unseen Families:** Các họ mã độc hoàn toàn mới, xuất hiện lần đầu trong giai đoạn 2014-2025.

* **Mục tiêu:** Chứng minh sức mạnh của phương pháp **PU Learning** trong việc phát hiện các biến thể mã độc chưa từng có tiền lệ (Zero-day/New families) so với các mẫu đã biết.



##  Trích dẫn

Dự án có kế thừa và phát triển từ công trình:

~~~
@inproceedings{na2020deep,
  title={Deep Generative Positive-Unlabeled Learning under Selection Bias},
  author={Na, Byeonghu and Kim, Hyungi and Song, Kyungwoo and Moon, Il-Chul},
  booktitle={Proceedings of the 29th ACM International Conference on Information & Knowledge Management},
  pages={1155--1164},
  year={2020}
}
~~~
