import numpy as np
import os
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Hàm hỗ trợ chuyển đổi sang Tensor
def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32).to(device)

def analysis(model):
    # 1. Cấu hình thiết bị
    device = next(model.parameters()).device # Lấy device hiện tại của model
    model.eval() # Chuyển sang chế độ đánh giá (tắt Dropout, BatchNorm cố định)

    # 2. Load dữ liệu (Giữ nguyên logic numpy)
    model_config = model.config 
    print(f"Loading data from: {model_config.data}")
    
    data_path = './pu_data/'+model_config.data+'.npz'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    data_load = np.load(data_path)
    x_tr_l = data_load['x_tr_l']
    x_tr_u = data_load['x_tr_u']
    y_tr_l = data_load['y_tr_l']
    y_tr_u = data_load['y_tr_u']
    
    # Val & Test load cho đủ thủ tục (code gốc có load)
    x_val = data_load['x_val']
    x_te = data_load['x_te']
    data_load.close()

    # 3. Preprocessing (Chuẩn hóa)
    # Logic chung cho normalization
    # Lấy mẫu kiểm tra
    min_val = x_tr_l.min()
    max_val = x_tr_l.max()

    print(f"Detected data range: [{min_val:.2f}, {max_val:.2f}]")

    # Nếu dữ liệu nằm trong khoảng âm (ví dụ -1 đến 1), ta mới scale
    if min_val < 0:
        print("Auto-normalizing to [0, 1]...")
        x_tr_l = (x_tr_l + 1.) / 2.
        x_tr_u = (x_tr_u + 1.) / 2.
        x_val = (x_val + 1.) / 2.
        x_te = (x_te + 1.) / 2.
    else:
        print("Data seems to be in [0, 1] already. Keeping as is.")

    # Logic riêng cho Convolution (CIFAR)
    if 'conv' in model_config.data:
        # [QUAN TRỌNG] PyTorch dùng (N, C, H, W)
        # Code gốc: reshape(-1, 3, 32, 32) -> transpose (N, H, W, C)
        # Code mới: Chỉ reshape, KHÔNG transpose
        print("Reshaping for Conv2d (N, C, H, W)...")
        x_tr_l = x_tr_l.reshape(-1, 3, 32, 32)
        x_tr_u = x_tr_u.reshape(-1, 3, 32, 32)
        x_val = x_val.reshape(-1, 3, 32, 32)
        x_te = x_te.reshape(-1, 3, 32, 32)

    # Logic riêng cho News
    # PyTorch tự ép kiểu lúc to_tensor nên không cần if 'news' cast thủ công ở đây

    # 4. Tạo biến 'o' (Observation Indicator)
    # Labeled data: o = [1, 0]
    # Unlabeled data: o = [0, 1]
    
    # Tạo numpy trước rồi convert sau
    o1 = np.hstack([np.ones((x_tr_l.shape[0], 1)), np.zeros((x_tr_l.shape[0], 1))])
    o2 = np.hstack([np.zeros((x_tr_u.shape[0], 1)), np.ones((x_tr_u.shape[0], 1))])
    o = np.concatenate([o1, o2], axis=0)

    # Ghép dữ liệu x
    x_combined = np.concatenate([x_tr_l, x_tr_u], axis=0)
    
    # Tạo nhãn màu để vẽ (y)
    # Code gốc dùng 0.5 cho labeled để phân biệt màu
    y_combined = np.concatenate([0.5 * np.ones_like(y_tr_l), y_tr_u], axis=0)

    print("Running t-SNE analysis...")
    # Gọi hàm vẽ cho h_y
    test_tsne(model, model_config, x_combined, y_combined, o, 'train_tsne_add_obs_h_y', mode='h_y', device=device)
    
    # Gọi hàm vẽ cho h_o
    test_tsne(model, model_config, x_combined, y_combined, o, 'train_tsne_add_obs_h_o', mode='h_o', device=device)
    print("Analysis done. Check output folder.")


def test_tsne(model, model_config, x, y, o, fname, makecolor=False, mode='h_y', device='cpu'):
    # Xử lý màu sắc (Giữ nguyên logic code gốc)
    if makecolor == False:
        color = y
    else:
        color = []
        for i in range(len(y)):
            if y[i] == 3: color.append('g')
            elif y[i] == 2: color.append('r')
            elif y[i] == 1: color.append('y')
            elif y[i] == -1: color.append('m')
            elif y[i] == -2: color.append('b')
            else: color.append('k') # Mặc định đen nếu lạ

    # --- ENCODING (Phần quan trọng nhất) ---
    # Vì dữ liệu có thể lớn, ta chạy theo batch để tránh tràn VRAM GPU
    batch_size = 1000 
    n_samples = x.shape[0]
    h_y_mus = []
    h_o_mus = []

    with torch.no_grad(): # Không tính gradient
        for i in range(0, n_samples, batch_size):
            # Cắt batch
            x_batch = x[i : i + batch_size]
            o_batch = o[i : i + batch_size]

            # Convert sang Tensor GPU
            x_batch_t = to_tensor(x_batch, device)
            o_batch_t = to_tensor(o_batch, device)

            # Forward qua Encoder
            # model_en trả về: mu_y, logvar_y, mu_o, logvar_o
            mu_y, _, mu_o, _ = model.model_en(x_batch_t, o_batch_t)

            # Đưa về CPU numpy và lưu lại
            h_y_mus.append(mu_y.cpu().numpy())
            h_o_mus.append(mu_o.cpu().numpy())

    # Nối lại thành array lớn
    h_y_mu = np.concatenate(h_y_mus, axis=0)
    h_o_mu = np.concatenate(h_o_mus, axis=0)

    # --- T-SNE ---
    tsne = TSNE(n_components=2, random_state=42) # random_state để kết quả cố định đẹp hơn

    if mode == 'h_y':
        print(f"Fitting t-SNE for {mode}...")
        trans = tsne.fit_transform(h_y_mu)
    elif mode == 'h_o':
        # Nếu chiều latent của o chỉ là 2 thì vẽ luôn, ko cần tsne
        if model_config['n_h_o'] == 2:
            trans = h_o_mu
        else:
            print(f"Fitting t-SNE for {mode}...")
            trans = tsne.fit_transform(h_o_mu)
    else:
        raise NotImplementedError()

    # --- PLOTTING ---
    plt.figure(figsize=(10, 10)) # Canvas to hơn chút cho đẹp
    plt.scatter(trans[:, 0], trans[:, 1], c=color, s=2, cmap='jet', alpha=0.6) # s=size điểm, alpha=độ trong suốt
    
    # Nếu dùng color map số (y) thì hiện thanh colorbar
    if makecolor == False:
        plt.colorbar()

    plt.title(f't-SNE visualization of {mode}')
    save_path = model_config['directory'] + fname + '.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")