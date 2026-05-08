import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# --- 1. TỰ ĐỘNG TÌM ĐƯỜNG DẪN model.py ---
found_path = False
# Quét tìm thư mục chứa model.py để thêm vào sys.path
for root, dirs, files in os.walk('/kaggle/input/'):
    if 'model.py' in files and '__pycache__' not in root:
        sys.path.append(root)
        print(f"Đã tìm thấy model.py tại: {root}")
        found_path = True
        break

if not found_path:
    print("Không tìm thấy model.py. Vui lòng kiểm tra lại Notebook Output!")
else:
    from model import VAEencoder, VAEdecoder, discriminator, classifier_o, classifier_pn, myPU
    from config import config

def predict_hack(model_pn, x):
    curr_x = x
    for layer in model_pn.layers:
        curr_x = layer(curr_x)
    return curr_x

# --- 2. HÀM ĐÁNH GIÁ PIPELINE ---
def run_evaluation_pipeline(checkpoint_dir, model_config, data_dir):
    print("Đang khởi tạo mô hình và khôi phục trọng số...")
    model_config['n_input'] = 925
    model_config['n_h_y'], model_config['n_h_o'] = 10, 2
    model_config['n_hidden_pn'] = [300, 300, 300, 300]
    
    model_en = VAEencoder(model_config)
    model_de = VAEdecoder(model_config)
    model_disc = discriminator(model_config)
    model_cl = classifier_o(model_config)
    model_pn = classifier_pn(model_config)
    opt = tf.keras.optimizers.Adam()
    
    model = myPU(model_config, model_en, model_de, model_disc, model_cl, model_pn, opt, opt, opt, opt, opt)
    checkpoint = tf.train.Checkpoint(model_en=model.model_en, model_de=model.model_de, 
                                     model_disc=model.model_disc, model_cl=model.model_cl, model_pn=model.model_pn)
    
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=100)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
    else:
        print("❌ Không tìm thấy Checkpoint!"); return [], []

    target_files = ["test_pl_and_pu_features_1.npz", "test_seen_pl_features_1.npz", "test_pure_unseen_features_1.npz"]
    feature_files = [os.path.join(data_dir, f) for f in target_files if os.path.exists(os.path.join(data_dir, f))]
    
    generated_csv_files = []
    metrics_list = []
    
    print(f"\n🔎 BẮT ĐẦU DỰ ĐOÁN 3 KỊCH BẢN...")
    for f_path in feature_files:
        base_name = os.path.basename(f_path)
        meta_path = os.path.join(data_dir, base_name.replace("_features_1.npz", "_metadata_1.csv"))
        prefix = base_name.replace("test_", "").replace("_features_1.npz", "").upper()

        if not os.path.exists(meta_path): continue

        # Load dữ liệu
        data = np.load(f_path)
        x_te, y_te = tf.cast(data['x_te'], tf.float32), data['y_te']
        dataset = tf.data.Dataset.from_tensor_slices((x_te, y_te)).batch(1000)

        # 1. Tính toán chỉ số tổng quát
        acc, pre, rec = model.accuracy(dataset)
        f1 = (2 * pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
        metrics_list.append({'Scenario': prefix, 'Accuracy': acc, 'Precision': pre, 'Recall': rec, 'F1-Score': f1})

        # 2. Dự đoán chi tiết cho từng mẫu
        y_pred_list = [np.where(predict_hack(model.model_pn, xb).numpy() > 0.5, 1, -1) for xb, _ in dataset]
        y_pred_final = np.concatenate(y_pred_list, axis=0).flatten()

        try:
            df_meta = pd.read_csv(meta_path, encoding='utf-8')
        except:
            df_meta = pd.read_csv(meta_path, encoding='latin1')
            
        df_meta['model_prediction'] = y_pred_final
        out_name = f"result_2013_{prefix.lower()}.csv"
        df_meta.to_csv(out_name, index=False)
        generated_csv_files.append(out_name)
        print(f"   Hoàn tất kịch bản: {prefix}")

    return generated_csv_files, metrics_list

# --- 3. BIỂU ĐỒ 1: CHỈ SỐ ĐÁNH GIÁ (MACRO METRICS) ---
def plot_macro_metrics(metrics_list):
    print("\nBIỂU ĐỒ 1: SO SÁNH CHỈ SỐ ĐÁNH GIÁ GIỮA CÁC KỊCH BẢN")
    df_metrics = pd.DataFrame(metrics_list)
    df_melted = df_metrics.melt(id_vars='Scenario', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(15, 7))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df_melted, x='Scenario', y='Score', hue='Metric', palette='viridis')
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(format(p.get_height(), '.3f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 9), 
                        textcoords='offset points', fontsize=9, fontweight='bold')

    plt.title('Hiệu năng mô hình trên 3 kịch bản kiểm tra (2013)', fontsize=16, pad=20)
    plt.ylim(0, 1.15)
    plt.ylabel('Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# --- 4. BIỂU ĐỒ 2: PHÂN TÍCH VÙNG XÁM (GRAY ZONE) ---
def analyze_gray_zone_detail(file_list):
    print("\nBIỂU ĐỒ 2: PHÂN TÍCH KHẢ NĂNG NHẬN DIỆN VÙNG XÁM (VT 4-12)")
    all_data = []
    for f in file_list:
        if os.path.exists(f):
            all_data.append(pd.read_csv(f))
    
    if not all_data: return
        
    df_combined = pd.concat(all_data, ignore_index=True).drop_duplicates(subset=['hash'])
    
    # Lọc Malware vùng xám
    df_gray = df_combined[(df_combined['label'] == 1) & 
                          (df_combined['vt_count'] >= 4) & 
                          (df_combined['vt_count'] <= 12)].copy()
    
    if df_gray.empty:
        print("Không tìm thấy dữ liệu Malware trong vùng VT 4-12."); return

    df_gray['vt_count'] = df_gray['vt_count'].astype(int)
    df_gray['Status'] = (df_gray['label'] == df_gray['model_prediction']).map({True: 'Đúng (TP)', False: 'Sai (FN)'})

    stats = df_gray.groupby(['vt_count', 'Status']).size().unstack(fill_value=0)
    for col in ['Đúng (TP)', 'Sai (FN)']:
        if col not in stats.columns: stats[col] = 0

    plt.figure(figsize=(14, 7))
    ax = stats[['Đúng (TP)', 'Sai (FN)']].plot(kind='bar', stacked=True, 
                                               color=['#27ae60', '#e74c3c'], 
                                               ax=plt.gca(), width=0.7)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            x, y = p.get_xy()
            idx = int(round(x))
            total = stats.loc[stats.index[idx]].sum()
            ax.text(x + p.get_width()/2, y + h/2, f'{int(h)}\n({(h/total)*100:.1f}%)', 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    plt.title('Độ nhạy của mô hình đối với Malware có VT_Count thấp (4-12)', fontsize=15, pad=20)
    plt.xlabel('Số lượng Engine phát hiện (VirusTotal Count)')
    plt.ylabel('Số lượng mẫu')
    plt.xticks(rotation=0)
    plt.legend(title='Kết quả')
    plt.tight_layout()
    plt.show()

# --- 5. THỰC THI ---
if __name__ == '__main__':
    # Đường dẫn Checkpoint và Dataset
    CKPT = '/kaggle/input/notebooks/nguyenminhkha1807/lamda-bias-pl-13-4-pu-12-50k/result/lamda_vae_pu_NON_SCAR_50k_vtcount4to12forPU_13forPL/Exp0/training_checkpoints/'
    DATA = '/kaggle/input/datasets/nguyenminhkha1807/kiem-tra-tap-test-3-kich-ban/'
    
    # Thực hiện quy trình
    csv_files, metrics = run_evaluation_pipeline(CKPT, config, DATA)
    
    if metrics:
        plot_macro_metrics(metrics)
        
    if csv_files:
        analyze_gray_zone_detail(csv_files)