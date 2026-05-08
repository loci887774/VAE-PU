import numpy as np
import os
import torch
import random
import sys

# Import các module 
from config import config
from train import train
from train_load import analysis  

# --- 1. CẤU HÌNH MÔI TRƯỜNG (KAGGLE & PYTORCH) ---
# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on device: {device}")

# Tắt cảnh báo linh tinh của Numpy/PyTorch nếu cần
import warnings
warnings.filterwarnings("ignore")

# --- 2. THIẾT LẬP THAM SỐ (Dạng SimpleNamespace) ---
idx = 0

# Tên file dữ liệu (Cậu sửa tên file cậu muốn chạy ở đây)
# Lưu ý: Trong train.py, đoạn load dữ liệu cậu nhớ trỏ đúng path '/kaggle/input/...' nhé
config.data = 'MNIST_35_val' 

# Cập nhật tham số dùng dấu chấm (.) thay vì ['key']
config.pi_pl = 0.01
config.pi_pu = 0.5 - config.pi_pl
config.pi_u = 1 - config.pi_pl

config.n_h_y = 10
config.n_h_o = 2
config.lr_pu = 3e-4
config.lr_pn = 1e-5

# Số Epoch 
config.num_epoch_pre = 100
config.num_epoch_step1 = 400
config.num_epoch_step_pn1 = 500
config.num_epoch_step_pn2 = 600
config.num_epoch_step2 = 500
config.num_epoch_step3 = 700
config.num_epoch = 800

config.n_hidden_cl = []
config.n_hidden_pn = [300, 300, 300, 300]

# Gán batch size
config.batch_size_l = 10
config.batch_size_u = 990
config.batch_size_l_pn = 10
config.batch_size_u_pn = 990

# Các hệ số Loss
config.alpha_gen = 0.1
config.alpha_disc = 0.1
config.alpha_gen2 = 3
config.alpha_disc2 = 3

# --- 3. HÀM SET SEED (Để kết quả không nhảy lung tung) ---
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ép PyTorch chạy đơn luồng xác định (sẽ chậm hơn chút nhưng kết quả giống hệt nhau)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(idx)

# --- 4. CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    print(f"\n>>> 🏁 BẮT ĐẦU HUẤN LUYỆN MODEL VỚI DATA: {config.data}")
    
    # Bước 1: Gọi hàm train
    # LƯU Ý SỐ 1: Hàm train(idx, config) của cậu phải trả về biến 'model' ở cuối cùng nhé!
    # LƯU Ý SỐ 2: config bây giờ là object, nên trong train.py cậu cũng phải dùng config.lr_pu (dấu chấm)
    trained_model = train(idx, config)

    # Bước 2: Lưu model trên Kaggle (Phải lưu vào /kaggle/working/)
    save_path = '/kaggle/working/final_model.pth'
    torch.save(trained_model.state_dict(), save_path)
    print(f"\n✅ Đã lưu model tại: {save_path}")

    print("\n>>> 📊 ĐANG VẼ BIỂU ĐỒ T-SNE & TÍNH METRICS (train_load.py)...")
    
    # Bước 3: Gọi 'thần thú' analysis
    # Nó sẽ lấy model vừa train, chạy lại trên tập test, vẽ hình và lưu vào folder ảnh
    analysis(trained_model)
    
    print("\n>>> 🎉 XONG PHIM! KIỂM TRA MỤC OUTPUT CỦA KAGGLE THOI!")