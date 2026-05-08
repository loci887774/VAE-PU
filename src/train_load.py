#%%writefile train_load.py 

import numpy as np
import time
import tensorflow as tf
from config import config
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

#tfe = tf.contrib.eager
#tf.compat.v1.enable_eager_execution()

def plot_virtual_vs_real(model, x_pl, x_u, y_u, fname='tsne_virtual_vs_real.png'):
    """
    Hàm nén chiều t-SNE và vẽ phân phối dữ liệu cho VAE-PU.
    Tách biệt dữ liệu thực tế (PL, PU, N) và mẫu ảo được sinh ra.
    """
    print("\n--- Đang tiến hành phân tích và vẽ Concept Drift (Virtual vs Real) ---")
    
    # 1. Sinh mẫu ảo từ mô hình (truyền vào mode từ config của model)
    mode = model.config.get('mode', 'near_o')
    _, _, _, x_pu_virtual, _ = model.generate(x_pl, x_u, mode=mode)

    # 2. Chuyển đổi Tensors sang NumPy arrays để tương thích với scikit-learn
    x_pl_np = x_pl.numpy() if hasattr(x_pl, 'numpy') else np.array(x_pl)
    x_u_np = x_u.numpy() if hasattr(x_u, 'numpy') else np.array(x_u)
    x_pu_virtual_np = x_pu_virtual.numpy() if hasattr(x_pu_virtual, 'numpy') else np.array(x_pu_virtual)
    y_u_np = y_u.numpy() if hasattr(y_u, 'numpy') else np.array(y_u).flatten()

    # 3. Phân tách tập Unlabeled thành PU và Negative thực tế dựa trên y_u
    # (Dùng y != 1 để bao quát cả trường hợp nhãn Negative là 0 hoặc -1)
    x_pu_real_np = x_u_np[y_u_np == 1]
    x_n_real_np = x_u_np[y_u_np != 1]

    # 4. Hợp nhất thành ma trận lớn để chạy t-SNE cùng một không gian
    X_all = np.vstack([x_pl_np, x_pu_real_np, x_n_real_np, x_pu_virtual_np])

    print(f"Bắt đầu chạy t-SNE nén {X_all.shape[1]} chiều xuống 2D trên {X_all.shape[0]} mẫu...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_all)

    # 5. Trích xuất lại tọa độ sau khi nén
    len_pl = len(x_pl_np)
    len_pu_real = len(x_pu_real_np)
    len_n_real = len(x_n_real_np)

    idx_pl_end = len_pl
    idx_pu_real_end = idx_pl_end + len_pu_real
    idx_n_real_end = idx_pu_real_end + len_n_real

    tsne_pl = X_tsne[:idx_pl_end]
    tsne_pu_real = X_tsne[idx_pl_end:idx_pu_real_end]
    tsne_n_real = X_tsne[idx_pu_real_end:idx_n_real_end]
    tsne_virtual = X_tsne[idx_n_real_end:]

    # 6. Scatter Plot
    plt.figure(figsize=(10, 8))

    # Vẽ Vòng tròn cam: Các mẫu Negative thực tế
    plt.scatter(tsne_n_real[:, 0], tsne_n_real[:, 1], 
                c='orange', marker='o', alpha=0.4, label='Real Negative (N)', edgecolors='w')

    # Vẽ Vòng tròn xanh lam: Các mẫu Positive-Unlabeled thực tế
    plt.scatter(tsne_pu_real[:, 0], tsne_pu_real[:, 1], 
                c='blue', marker='o', alpha=0.5, label='Real PU (Malware in U)', edgecolors='w')

    # Vẽ Vòng tròn đỏ: Các mẫu Positive-Labeled thực tế
    plt.scatter(tsne_pl[:, 0], tsne_pl[:, 1], 
                c='red', marker='o', alpha=0.8, label='Real Positive-Labeled (PL)', edgecolors='w')

    # Vẽ Hình vuông đen: Các mẫu ảo được sinh ra (Virtual)
    plt.scatter(tsne_virtual[:, 0], tsne_virtual[:, 1], 
                c='black', marker='s', alpha=0.7, label='Virtual Samples (Generated)', edgecolors='w')

    plt.title('t-SNE Visualization: Virtual vs Real Samples (Concept Drift)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Hoàn tất! Đã lưu biểu đồ tại: {fname}\n")


def analysis(model):
    # load PU data (x_tr_l = train labeled data, x_tr_u = train unlabeled data, x_te = test data, y_te = test label)
    model_config = model.config
    #path_kaggle = '/kaggle/input/datasets/nguyenminhkha1807/lamda2013-vae-pu-bias-extreme-vtcount17/'
    path_kaggle = '/kaggle/input/datasets/nguyenminhkha1807/lamda-50k/'
    
    data_load = np.load(path_kaggle + model_config['data']+'.npz')
    x_tr_l = data_load['x_tr_l']
    x_tr_u = data_load['x_tr_u']
    y_tr_l = data_load['y_tr_l']
    y_tr_u = data_load['y_tr_u']

    x_val = data_load['x_val']
    y_val = data_load['y_val']

    x_te = data_load['x_te']
    y_te = data_load['y_te']
    data_load.close()

    if 'lamda' in model_config['data']:
        # Dữ liệu đã là 0-1 rồi, chỉ cần đảm bảo là float32 cho TensorFlow
        x_tr_l = tf.cast(x_tr_l, tf.float32)
        x_tr_u = tf.cast(x_tr_u, tf.float32)
        x_val = tf.cast(x_val, tf.float32)
        x_te = tf.cast(x_te, tf.float32)

    if 'MNIST' in model_config['data']:
        x_tr_l = (x_tr_l + 1.) / 2.
        x_tr_u = (x_tr_u + 1.) / 2.
        x_val = (x_val + 1.) / 2.
        x_te = (x_te + 1.) / 2.

    if 'conv' in model_config['data']:
        x_tr_l = (x_tr_l + 1.) / 2.
        x_tr_u = (x_tr_u + 1.) / 2.
        x_val = (x_val + 1.) / 2.
        x_te = (x_te + 1.) / 2.

    if 'conv' in model_config['data']:
        # CIFAR
        x_tr_l = np.transpose(x_tr_l.reshape(-1, 3, 32, 32), (0, 2, 3, 1))
        x_tr_u = np.transpose(x_tr_u.reshape(-1, 3, 32, 32), (0, 2, 3, 1))
        x_val = np.transpose(x_val.reshape(-1, 3, 32, 32), (0, 2, 3, 1))
        x_te = np.transpose(x_te.reshape(-1, 3, 32, 32), (0, 2, 3, 1))

    if 'news' in model_config['data']:
        x_tr_l = tf.cast(x_tr_l, tf.float32)
        x_tr_u = tf.cast(x_tr_u, tf.float32)
        x_val = tf.cast(x_val, tf.float32)
        x_te = tf.cast(x_te, tf.float32)

    # clustering of h_y
    o1 = tf.concat([tf.ones([x_tr_l.shape[0], 1]), tf.zeros([x_tr_l.shape[0], 1])], axis=1)
    o2 = tf.concat([tf.zeros([x_tr_u.shape[0], 1]), tf.ones([x_tr_u.shape[0], 1])], axis=1)
    o = tf.concat([o1, o2], axis=0)
    
    # Các hàm vẽ t-SNE không gian tiềm ẩn (ẩn) mặc định của code cũ
    test_tsne(model, model_config, np.concatenate([x_tr_l, x_tr_u], axis=0), np.concatenate([0.5*np.ones_like(y_tr_l[:]), y_tr_u[:]], axis=0), o, 'train_tsne_add_obs_h_y', mode='h_y')
    test_tsne(model, model_config, np.concatenate([x_tr_l, x_tr_u], axis=0), np.concatenate([0.5*np.ones_like(y_tr_l[:]), y_tr_u[:]], axis=0), o, 'train_tsne_add_obs_h_o', mode='h_o')

    # ---> GỌI HÀM VẼ CONCEPT DRIFT (Virtual vs Real) Ở ĐÂY <---
    fname_virtual_vs_real = model_config['directory'] + 'tsne_virtual_vs_real.png'
    plot_virtual_vs_real(model, x_tr_l, x_tr_u, y_tr_u, fname=fname_virtual_vs_real)


def test_tsne(model, model_config, x, y, o, fname, makecolor=False, mode='h_y'):

    if makecolor == False:
        current_color = y
    else:
        color_list = []
        for i in range(len(y)):
            if y[i] == 3: color_list.append('g')
            elif y[i] == 2: color_list.append('r')
            elif y[i] == 1: color_list.append('y')
            elif y[i] == -1: color_list.append('m')
            elif y[i] == -2: color_list.append('b')
            else: color_list.append('gray')
            current_color = color_list

    h_y_mu, h_y_log_sig_sq, h_o_mu, h_o_log_sig_sq = model.model_en.encode(x, o)

    tsne = TSNE(n_components=2)

    if mode == 'h_y':
        trans = tsne.fit_transform(h_y_mu)
    elif mode == 'h_o':
        if model_config['n_h_o'] == 2:
            trans = h_o_mu
        else:
            trans = tsne.fit_transform(h_o_mu)
    else:
        NotImplementedError()

    plt.figure()
    plt.scatter(trans[:, 0], trans[:, 1], c=current_color, s=0.1)

    plt.savefig(model_config['directory'] + fname + '.png')
    plt.close()