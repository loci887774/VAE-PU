import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys
import re
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# --- 1. SETUP MÔI TRƯỜNG & IMPORT MODEL ---
for root, dirs, files in os.walk('/kaggle/input/'):
    if 'model.py' in files and '__pycache__' not in root:
        sys.path.append(root)
        break

from model import VAEencoder, VAEdecoder, discriminator, classifier_o, classifier_pn, myPU
from config import config

# --- 2. HÀM KHỞI TẠO MÔ HÌNH ---
def init_model_for_eval(checkpoint_dir, model_config):
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
        return model
    return None

def predict_hack(model_pn, x):
    curr_x = x
    for layer in model_pn.layers:
        curr_x = layer(curr_x)
    return curr_x

# --- 3. PIPELINE ĐÁNH GIÁ & THỐNG KÊ CHI TIẾT ---
def run_comprehensive_evaluation(checkpoint_dir, model_config, data_dir):
    model = init_model_for_eval(checkpoint_dir, model_config)
    if not model: return None

    # Tìm tất cả file features
    feature_files = glob.glob(os.path.join(data_dir, "*_features.npz"))
    all_results = []

    print(f" Tìm thấy {len(feature_files)} kịch bản. Bắt đầu phân tích chi tiết...")

    for f_path in feature_files:
        f_name = os.path.basename(f_path)
        year_match = re.search(r'20\d{2}', f_name)
        year = year_match.group(0) if year_match else "Unknown"
        if year == "2015": continue 

        group_type = 'Seen' if 'seen' in f_name.lower() and 'unseen' not in f_name.lower() else 'Unseen'
        meta_path = f_path.replace("_features.npz", "_metadata.csv")

        # Load Features & Predict
        data = np.load(f_path)
        x_te, y_te = tf.cast(data['x_te'], tf.float32), data['y_te']
        dataset = tf.data.Dataset.from_tensor_slices((x_te, y_te)).batch(1000)

        # Tính Metrics chính xác từ mô hình
        acc, pre, rec = model.accuracy(dataset)
        f1 = (2 * pre * rec) / (pre + rec) if (pre + rec) > 0 else 0

        # Thống kê họ mã độc từ Metadata
        if os.path.exists(meta_path):
            try:
                df_meta = pd.read_csv(meta_path, encoding='utf-8')
            except:
                df_meta = pd.read_csv(meta_path, encoding='latin1')
            
            # Dự đoán nhãn để lọc malware phát hiện được (tùy chọn, ở đây thống kê trên label thực)
            malware_df = df_meta[df_meta['label'] == 1]
            num_fams = malware_df['family'].nunique()
            top_fams = malware_df['family'].value_counts().head(5).index.tolist()
        else:
            num_fams, top_fams = 0, ["N/A"]

        all_results.append({
            'Year': year,
            'Type': group_type,
            'Acc': acc,
            'Pre': pre,
            'Rec': rec,
            'F1': f1,
            'Num_Fams': num_fams,
            'Top_5': ", ".join(top_fams)
        })
        print(f"  Xử lý xong {year} - {group_type}")

    return pd.DataFrame(all_results).sort_values(['Type', 'Year'])

# --- 4. VẼ BIỂU ĐỒ SUBPLOTS (SEEN & UNSEEN) ---
def plot_concept_drift_subplots(df_plot):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    def plot_on_axis(ax, data_type, title_color):
        subset = df_plot[df_plot['Type'] == data_type]
        if subset.empty: return

        ax.plot(subset['Year'], subset['Acc'], marker='o', label='Accuracy', color='#1f77b4', linewidth=1.5)
        ax.plot(subset['Year'], subset['Pre'], marker='s', label='Precision', color='#e377c2', linewidth=1.5)
        ax.plot(subset['Year'], subset['Rec'], marker='^', label='Recall', color='#2ca02c', linewidth=1.5)
        ax.plot(subset['Year'], subset['F1'],  marker='d', label='F1-Score', color='red', linewidth=3)

        ax.set_title(f'BIỂU ĐỒ CONCEPT DRIFT - NHÓM: {data_type.upper()}', fontsize=14, fontweight='bold', color=title_color)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='lower left', frameon=True, shadow=True)
        ax.set_ylabel('Giá trị Metric')

    plot_on_axis(ax1, 'Seen', 'darkblue')
    plot_on_axis(ax2, 'Unseen', 'darkred')

    plt.xlabel('Năm đánh giá (Thời gian)', fontsize=12)
    plt.tight_layout()
    plt.show()

# --- 5. THỰC THI TỔNG THỂ ---
if __name__ == '__main__':
    CKPT = '/kaggle/input/notebooks/nguyenminhkha1807/lamda-bias-pl-13-4-pu-12-50k/result/lamda_vae_pu_NON_SCAR_50k_vtcount4to12forPU_13forPL/Exp0/training_checkpoints/'
    DATA_PATH = '/kaggle/input/datasets/nguyenminhkha1807/family-malware/'
    
    df_results = run_comprehensive_evaluation(CKPT, config, DATA_PATH)
    
    if df_results is not None:
        # Vẽ biểu đồ Subplots
        plot_concept_drift_subplots(df_results)
        
        # In bảng thống kê chi tiết
        print("\n" + "="*110)
        print(f"{'Năm':<8} | {'Phân loại':<10} | {'F1-Score':<10} | {'Số họ':<8} | {'Top 5 họ mã độc'}")
        print("-"*110)
        
        df_table = df_results.sort_values(['Year', 'Type'])
        for _, row in df_table.iterrows():
            print(f"{row['Year']:<8} | {row['Type']:<10} | {row['F1']:<10.4f} | {row['Num_Fams']:<8} | {row['Top_5']}")
        print("="*110)
        
        # Lưu kết quả
        df_table.to_csv('final_concept_drift_seen_unseen.csv', index=False)
        print(" Đã lưu bảng thống kê vào file: final_concept_drift_seen_unseen.csv")