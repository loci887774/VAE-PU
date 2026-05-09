import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. TỰ ĐỘNG TÌM ĐƯỜNG DẪN model.py ---
found_path = False
for root, dirs, files in os.walk('/kaggle/input/'):
    if 'model.py' in files and '__pycache__' not in root:
        sys.path.append(root)
        print(f"🎯 Đã tìm thấy model.py tại: {root}")
        found_path = True
        break

from model import VAEencoder, VAEdecoder, discriminator, classifier_o, classifier_pn, myPU
from config import config

# --- HÀM TEST TRẢ VỀ KẾT QUẢ ---
def run_test_on_file(checkpoint_dir, model_config, file_path, model_instance):
    try:
        data = np.load(file_path)
        x_new = data['X_test'] if 'X_test' in data else data['x_te']
        y_new = data['y_test'] if 'y_test' in data else data['y_te']
        
        x_new = tf.cast(x_new, tf.float32)
        batch_size = int(model_config.get('batch_size_test', 100))
        dataset_new = tf.data.Dataset.from_tensor_slices((x_new, y_new)).batch(batch_size)

        acc, precision, recall = model_instance.accuracy(dataset_new)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Trích xuất năm từ tên file bằng Regex
        file_name = os.path.basename(file_path)
        year_match = re.search(r'20\d{2}', file_name)
        year = year_match.group(0) if year_match else file_name

        return {
            'Year': year,
            'Accuracy': round(acc, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4)
        }
    except Exception as e:
        print(f"❌ Lỗi khi test file {file_path}: {e}")
        return None

# --- HÀM VẼ BIỂU ĐỒ TỔNG QUÁT ---
def plot_overall_concept_drift(df):
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    # Vẽ các đường metric
    plt.plot(df['Year'], df['Accuracy'], marker='o', label='Accuracy', color='#1f77b4', linewidth=1.5)
    plt.plot(df['Year'], df['Precision'], marker='s', label='Precision', color='#e377c2', linewidth=1.5)
    plt.plot(df['Year'], df['Recall'], marker='^', label='Recall', color='#2ca02c', linewidth=1.5)
    plt.plot(df['Year'], df['F1-Score'], marker='d', label='F1-Score', color='red', linewidth=3) # Highlight F1-Score

    plt.title('BIỂU ĐỒ ĐÁNH GIÁ CONCEPT DRIFT TỔNG QUÁT (2013-2025)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Năm đánh giá (Thời gian)', fontsize=12)
    plt.ylabel('Giá trị Metric (0.0 - 1.0)', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left', frameon=True, shadow=True, fontsize=11)
    
    # Thêm chú thích giá trị F1 lên biểu đồ
    for i, txt in enumerate(df['F1-Score']):
        plt.annotate(f"{txt:.3f}", (df['Year'].iloc[i], df['F1-Score'].iloc[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', 
                     fontsize=9, fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig('overall_concept_drift_chart.png')
    plt.show()

if __name__ == '__main__':
    CHECKPOINT_DIR = '/kaggle/input/notebooks/nguyenminhkha1807/lamda-bias-pl-13-4-pu-12-50k/result/lamda_vae_pu_NON_SCAR_50k_vtcount4to12forPU_13forPL/Exp0/training_checkpoints/'
    DATASET_DIR = '/kaggle/input/datasets/nguyenminhkha1807/test-concept-drift-auto/'
    
    # 2. THIẾT LẬP MODEL
    print("⚙️ Đang khởi tạo mô hình...")
    config['n_input'] = 925
    config['n_h_y'], config['n_h_o'] = 10, 2
    config['n_hidden_pn'] = [300, 300, 300, 300]
    
    model_en = VAEencoder(config)
    model_de = VAEdecoder(config)
    model_disc = discriminator(config)
    model_cl = classifier_o(config)
    model_pn = classifier_pn(config)
    opt = tf.keras.optimizers.Adam()
    
    model_pu = myPU(config, model_en, model_de, model_disc, model_cl, model_pn, opt, opt, opt, opt, opt)
    
    checkpoint = tf.train.Checkpoint(model_en=model_pu.model_en, model_de=model_pu.model_de, 
                                     model_disc=model_pu.model_disc, model_cl=model_pu.model_cl, 
                                     model_pn=model_pu.model_pn)
    
    manager = tf.train.CheckpointManager(checkpoint, directory=CHECKPOINT_DIR, max_to_keep=100)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(f"✅ Đã khôi phục checkpoint: {manager.latest_checkpoint}")
    else:
        print("❌ Không tìm thấy checkpoint!")
        sys.exit()

    # 3. CHẠY TEST TỰ ĐỘNG
    file_list = sorted(glob.glob(os.path.join(DATASET_DIR, "*.npz")))
    
    if not file_list:
        print(f"❌ Không tìm thấy file dữ liệu tại {DATASET_DIR}")
    else:
        all_results = []
        for file_path in file_list:
            res = run_test_on_file(CHECKPOINT_DIR, config, file_path, model_pu)
            if res: all_results.append(res)
        
        df_final = pd.DataFrame(all_results)
        
        # Sắp xếp theo năm để vẽ biểu đồ không bị rối
        df_final = df_final.sort_values('Year')

        print("\n" + "="*60)
        print("📊 TỔNG HỢP KẾT QUẢ CONCEPT DRIFT")
        print("="*60)
        print(df_final.to_string(index=False))
        
        # 4. VẼ BIỂU ĐỒ VÀ LƯU KẾT QUẢ
        plot_overall_concept_drift(df_final)
        df_final.to_csv('concept_drift_summary_full.csv', index=False)
        print("\n💾 Kết quả đã được lưu thành file CSV và ảnh PNG.")