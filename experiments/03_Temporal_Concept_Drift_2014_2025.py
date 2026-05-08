#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys
import glob # Thư viện để tìm file theo mẫu

# --- 1. TỰ ĐỘNG TÌM ĐƯỜNG DẪN model.py ---
found_path = False
for root, dirs, files in os.walk('/kaggle/input/notebooks/nguyenminhkha1807/lamda-bias-pl-13-4-pu-12-50k/'):
    if 'model.py' in files:
        sys.path.append(root)
        print(f"Đã tìm thấy model.py tại: {root}")
        found_path = True
        break

from model import VAEencoder, VAEdecoder, discriminator, classifier_o, classifier_pn, myPU
from config import config

# --- HÀM TEST ĐÃ NÂNG CẤP (TRẢ VỀ KẾT QUẢ) ---
def run_test_on_file(checkpoint_dir, model_config, file_path, model_instance):
    try:
        data = np.load(file_path)
        # Kiểm tra key trong file npz (X_test hoặc x_te tùy cậu lưu)
        x_new = data['X_test'] if 'X_test' in data else data['x_te']
        y_new = data['y_test'] if 'y_test' in data else data['y_te']
        
        x_new = tf.cast(x_new, tf.float32)
        batch_size = int(model_config.get('batch_size_test', 100))
        dataset_new = tf.data.Dataset.from_tensor_slices((x_new, y_new)).batch(batch_size)

        acc, precision, recall = model_instance.accuracy(dataset_new)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'Year/File': os.path.basename(file_path),
            'Accuracy': round(acc, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4)
        }
    except Exception as e:
        print(f"Lỗi khi test file {file_path}: {e}")
        return None

if __name__ == '__main__':
    # 1. CẤU HÌNH ĐƯỜNG DẪN /kaggle/input/notebooks/nguyenminhkha1807/lamda-bias-pl-13-4-pu-12-50k/result/lamda_vae_pu_NON_SCAR_50k_vtcount4to12forPU_13forPL/Exp0/loss_gen.png
    CHECKPOINT_DIR = '/kaggle/input/notebooks/nguyenminhkha1807/lamda-bias-pl-13-4-pu-12-50k/result/lamda_vae_pu_NON_SCAR_50k_vtcount4to12forPU_13forPL/Exp0/training_checkpoints/'
    DATASET_DIR = '/kaggle/input/datasets/nguyenminhkha1807/test-concept-drift-auto/' # Thư mục chứa 12 file
    
    # 2. THIẾT LẬP MODEL (CHỈ LÀM 1 LẦN)
    print("Đang khởi tạo mô hình")
    config['n_input'] = 925
    config['n_h_y'] = 10
    config['n_h_o'] = 2
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
        print(f"Đã khôi phục checkpoint: {manager.latest_checkpoint}")
    else:
        print("Không tìm thấy checkpoint!")
        sys.exit()

    # 3. TÌM TẤT CẢ FILE TEST TRONG THƯ MỤC
    # Tìm tất cả file .npz, sắp xếp theo tên để chạy từ 2013 -> 2025
    file_list = sorted(glob.glob(os.path.join(DATASET_DIR, "*.npz")))
    
    if not file_list:
        print(f"Không tìm thấy file .npz nào trong {DATASET_DIR}")
    else:
        print(f"Tìm thấy {len(file_list)} file để test. Bắt đầu...")
        
        all_results = []
        for file_path in file_list:
            print(f"⏳ Đang test: {os.path.basename(file_path)}...")
            res = run_test_on_file(CHECKPOINT_DIR, config, file_path, model_pu)
            if res:
                all_results.append(res)
        
        # 4. IN BẢNG TỔNG HỢP KẾT QUẢ
        df_final = pd.DataFrame(all_results)
        print("\n" + "="*50)
        print("BẢNG TỔNG HỢP KẾT QUẢ CONCEPT DRIFT XUYÊN THỜI GIAN")
        print("="*50)
        print(df_final.to_string(index=False))
        print("="*50)
        
        # Lưu file csv để cậu tải về máy làm biểu đồ trong luận văn
        df_final.to_csv('concept_drift_summary.csv', index=False)
        print("Đã lưu kết quả vào file: concept_drift_summary.csv")