import numpy as np
import time
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from itertools import cycle # Để thay thế .repeat() của TF
from model import VAEEncoder, VAEDecoder, Discriminator, ClassifierO, ClassifierPN, MyPU
# from config import config # Nếu cậu dùng file config riêng
import matplotlib.pyplot as plt
import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

def plotLoss(lst, fname, title="Training Process", y_label="Loss Value"):
    plt.figure(figsize=(10, 6)) # Tăng kích thước chút cho thoáng
    
    # Vẽ dữ liệu
    # Dùng range(1, ...) để epoch bắt đầu từ 1
    plt.plot(range(1, len(lst)+1), lst, label=y_label, linewidth=1.5)

    # --- XỬ LÝ TRỤC Y (PHẦN QUAN TRỌNG NHẤT) ---
    plt.yscale('log') 
    
    # 1. Định dạng lại nhãn trục Y (tắt số mũ)
    # Hàm này sẽ chuyển đổi số (x) thành chuỗi dạng gọn nhất ({:g})
    # Ví dụ: 0.0001 thay vì 10^-4, 100 thay vì 10^2
    def format_func(x, pos):
        return '{:g}'.format(x)
        
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    
    # 2. Tùy chỉnh lưới (Grid) để chi tiết hơn
    # Hiện lưới cho cả các mốc chính (major) và mốc phụ (minor)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    
    # Nếu muốn hiện cả nhãn số cho các dòng kẻ phụ (nếu khoảng cách log quá lớn)
    # ax.yaxis.set_minor_formatter(FuncFormatter(format_func)) 

    # --- NHÃN VÀ TIÊU ĐỀ ---
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(f'{y_label} (Log scale)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Thêm legend và lưu file
    plt.legend()
    plt.tight_layout() # Tự động căn lề cho đẹp, không bị cắt chữ
    plt.savefig(fname, dpi=150) # Tăng dpi để ảnh nét hơn
    plt.close()



def train(num_exp, model_config, pretrain=True):
    # 1. Cấu hình thiết bị (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_config.directory = './result/'+model_config.data +'/Exp' + str(num_exp) + '/'
    os.makedirs(model_config.directory, exist_ok=True)

    # 2. Load dữ liệu (Giữ nguyên logic numpy)
    # 2. Load dữ liệu (Đoạn code sửa lỗi đường dẫn)
    print(f"Loading data: {model_config.data}")
    
    # Tên file cần tìm
    file_name = f"{model_config.data}.npz"
    
    # Danh sách các chỗ có thể chứa file (Ưu tiên Kaggle trước)
    possible_paths = [
        f"/kaggle/input/{model_config.data}/{file_name}",       # Kaggle: Nếu tên dataset trùng tên file
        f"/kaggle/input/mnist-pu-dataset/{file_name}",         # Kaggle: Nếu tên dataset là mnist-pu-dataset
        f"/kaggle/input/{file_name.lower().replace('.npz','')}/{file_name}", # Kaggle: case thường
        f"./pu_data/{file_name}",                               # Máy cá nhân (Local)
        f"./{file_name}"                                        # Ngay tại thư mục hiện hành
    ]
    
    final_path = None
    
    # Vòng lặp đi tìm file
    for path in possible_paths:
        if os.path.exists(path):
            final_path = path
            break
            
    # Nếu tìm trong list trên không thấy, quét toàn bộ thư mục input của Kaggle
    if final_path is None and os.path.exists("/kaggle/input"):
        print("Searching in /kaggle/input/...")
        for root, dirs, files in os.walk("/kaggle/input"):
            if file_name in files:
                final_path = os.path.join(root, file_name)
                break
    
    if final_path is None:
        raise FileNotFoundError(f"❌ Không tìm thấy file {file_name} ở đâu cả! Hãy kiểm tra lại tên Dataset trên Kaggle.")
    
    print(f"✅ Found data at: {final_path}")
    data_load = np.load(final_path)
    x_tr_l = data_load['x_tr_l']
    x_tr_u = data_load['x_tr_u']
    y_tr_l = data_load['y_tr_l'] # Không dùng nhưng vẫn load
    y_tr_u = data_load['y_tr_u']

    x_val = data_load['x_val']
    y_val = data_load['y_val']
    # Cắt nhỏ val như code gốc
    # x_val = x_tr_l[:5] 
    # y_val = y_tr_l[:5]

    x_te = data_load['x_te']
    y_te = data_load['y_te']
    data_load.close()

    # Preprocessing (Chuyển sang Tensor luôn)
    # Chỉ chuẩn hóa nếu là MNIST hoặc conv, còn lamda thì KỆ NÓ
    # if 'MNIST' in model_config['data'] or 'conv' in model_config['data']:
    #     print("Normalizing data from [-1, 1] to [0, 1]...")
    #     x_tr_l = (x_tr_l + 1.) / 2.
    #     x_tr_u = (x_tr_u + 1.) / 2.
    #     x_val = (x_val + 1.) / 2.
    #     x_te = (x_te + 1.) / 2.
    # elif 'lamda' in model_config['data']:
    #     print("Data is 'lamda'. Assuming range is already [0, 1]. Skipping normalization.")
    
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
    
    # Chuyển numpy -> torch tensor -> đưa lên device
    # Lưu ý: Dữ liệu lớn thì không nên đưa hết lên GPU ngay, nhưng nếu VRAM đủ thì đưa lên cho nhanh
    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32)

    x_tr_l = to_tensor(x_tr_l)
    x_tr_u = to_tensor(x_tr_u)
    x_val = to_tensor(x_val)
    y_val = to_tensor(y_val) # Label thường là float hoặc long tùy hàm loss, ở đây binary nên float ok
    x_te = to_tensor(x_te)
    y_te = to_tensor(y_te)

    # Dữ liệu demo để vẽ ảnh
    x_demo = x_tr_l[:int(model_config.batch_size_l)].to(device)
    x_demo_u = x_tr_u[:int(model_config.batch_size_u)].to(device)

    # 3. Tạo DataLoader (Thay thế tf.data.Dataset)
    # TensorDataset bọc dữ liệu lại
    ds_pl = TensorDataset(x_tr_l) 
    ds_u = TensorDataset(x_tr_u)
    ds_val = TensorDataset(x_val, y_val)
    ds_test = TensorDataset(x_te, y_te)

    # DataLoader giúp batching và shuffling
    dl_pl = DataLoader(ds_pl, batch_size=int(model_config.batch_size_l), shuffle=True, drop_last=True)
    dl_u = DataLoader(ds_u, batch_size=int(model_config.batch_size_u), shuffle=True, drop_last=True)
    # dl_val = DataLoader(ds_val, batch_size=int(model_config.batch_size_val), shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=int(model_config.batch_size_val), shuffle=False)
    #dl_test = DataLoader(ds_test, batch_size=int(model_config.batch_size_test), shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=int(model_config.batch_size_test), shuffle=False)


    # 4. Khởi tạo Model và Optimizer
    print("Initializing models (Khởi tạo Model)...")
    # Khởi tạo các mạng con
    m_en = VAEEncoder(model_config).to(device)
    m_de = VAEDecoder(model_config).to(device)
    m_disc = Discriminator(model_config).to(device)
    m_cl = ClassifierO(model_config).to(device)
    m_pn = ClassifierPN(model_config).to(device)

    # Khởi tạo Optimizer (PyTorch dùng torch.optim)
    opt_en = torch.optim.Adam(m_en.parameters(), lr=model_config.lr_pu)
    opt_de = torch.optim.Adam(m_de.parameters(), lr=model_config.lr_pu)
    opt_disc = torch.optim.Adam(m_disc.parameters(), lr=model_config.lr_disc, betas=(model_config.beta1, model_config.beta2))
    opt_cl = torch.optim.Adam(m_cl.parameters(), lr=model_config.lr_pu, weight_decay=1e-5)
    opt_pn = torch.optim.Adam(m_pn.parameters(), lr=model_config.lr_pn, weight_decay=1e-5)

    # Gom tất cả vào class myPU (Class này chứa logic train_step, check, generate...)
    model = MyPU(model_config, m_en, m_de, m_disc, m_cl, m_pn, 
                 opt_en, opt_de, opt_disc, opt_cl, opt_pn)

    # Logging setup
    with open(model_config.directory + 'log.txt', 'w') as log:
        log.write("config\n" + str(model_config) + '\n')
    with open(model_config.directory + 'log_PN.txt', 'w') as log2:
        log2.write("config\n" + str(model_config) + '\n')

    # Các biến lưu lịch sử loss
    lstLoss1, lstLoss2, lstLoss3, lstLoss4, lstLoss5 = [], [], [], [], []
    lstAcc, lstVal = [], []
    preLoss1, preLoss2 = [], []

    lsttime1, lsttime2 = [], []

    # ================= PRE-TRAIN =================
    if pretrain:
        print("Starting Pre-train...")
        for epoch in range(1, model_config.num_epoch_pre + 1):
            print(f'[PRE-TRAIN] Exp: {num_exp} / Epoch: {epoch}')
            start_time = time.time()
            lst_1, lst_2 = [], []

            # Mẹo thay thế .repeat() của TF:
            # dl_pl (ít dữ liệu) sẽ được wrap bởi cycle() để lặp đi lặp lại
            # dl_u (nhiều dữ liệu) sẽ quyết định độ dài epoch
            # Zip sẽ lấy cặp (x_pl, x_u)
            for (x_pl_batch,), (x_u_batch,) in zip(cycle(dl_pl), dl_u):
            #for (x_pl_batch,), (x_u_batch,) in zip(dl_pl, dl_u):
                # Đưa batch lên GPU
                x_pl_batch, x_u_batch = x_pl_batch.to(device), x_u_batch.to(device)

                l1 = model.pretrain(x_pl_batch, x_u_batch)
                lst_1.append(l1)

                if model_config.bool_pn_pre:
                    l2 = model.train_step_pn_pre(x_pl_batch, x_u_batch)
                    lst_2.append(l2)

            # Logging pretrain
            avg_l1 = sum(lst_1) / len(lst_1)
            preLoss1.append(avg_l1)
            
            if model_config.bool_pn_pre:
                avg_l2 = sum(lst_2) / len(lst_2)
                preLoss2.append(avg_l2)
            
            if epoch % 10 == 0:
                print(f'[PRE-TRAIN] Epoch {epoch} finished in {time.time() - start_time:.2f}s. Loss1: {avg_l1:.4f}')

        plotLoss(preLoss1, 
                model_config.directory + 'loss_pretrain.png', 
                title="Pre-training VAE Loss", 
                y_label="Reconstruction Loss")
        if model_config.bool_pn_pre:
            plotLoss(preLoss2, 
                    model_config.directory + 'loss_pretrain_pn.png', 
                    title="Pre-training PN Loss", 
                    y_label="Risk Value")
                
        print('PRE-TRAIN finish!')
        
        # Tìm Prior (Cần chuyển x_tr_l, x_tr_u lên GPU trong hàm findPrior nếu cần)
        # Lưu ý: x_tr_l đang ở CPU (tensor), nếu hàm findPrior tính toán nặng thì nên batching hoặc đưa lên GPU
        model.findPrior(x_tr_l.to(device), x_tr_u.to(device))
        np.savez(model_config.directory +'prior', mu=model.mu.cpu().numpy(), var=model.var.cpu().numpy())

# ================= MAIN TRAINING =================
    print("Starting Main Training...")
    
    # --- ĐỒNG HỒ TỔNG ---
    total_start_time = time.time()
    
    lsttime1, lsttime2 = [], []

    for epoch in range(1, model_config.num_epoch + 1):
        # --- ĐỒNG HỒ EPOCH ---
        epoch_start_time = time.time()
        
        lst_1, lst_2, lst_3, lst_4, lst_5 = [], [], [], [], []

        # Logic Zip và Cycle tương tự Pretrain
        # !!!!!!! Trước vòng lặp epoch, lấy sẵn 1 mẫu check !!!!!!!
        # fixed_pl, fixed_u = next(iter(dl_pl)), next(iter(dl_u))
        # fixed_pl, fixed_u = fixed_pl[0].to(device), fixed_u[0].to(device)

        for (x_pl_batch,), (x_u_batch,) in zip(cycle(dl_pl), dl_u):
        #for (x_pl_batch,), (x_u_batch,) in zip(dl_pl, dl_u):
            x_pl_batch, x_u_batch = x_pl_batch.to(device), x_u_batch.to(device)

            # Step 3: Train Disc & VAE
            if epoch <= model_config.num_epoch_step3:
                if not (model_config.num_epoch_step1 < epoch <= model_config.num_epoch_step2):
                    l3 = model.train_step_disc(x_pl_batch, x_u_batch)
                    lst_3.append(l3)
                    
                    l1, l2, l4 = model.train_step_vae(x_pl_batch, x_u_batch, epoch)
                    lst_1.append(l1)
                    lst_2.append(l2)
                    lst_4.append(l4)

            # Step PN: Train Classifier PN
            if epoch > model_config.num_epoch_step1:
                if not (model_config.num_epoch_step_pn1 < epoch <= model_config.num_epoch_step_pn2):
                    l5 = model.train_step_pn(x_pl_batch, x_u_batch)
                    lst_5.append(l5)
            
        # --- KẾT THÚC VÒNG LẶP BATCH ---

        # Logging định kỳ vào file text
        if epoch <= model_config.num_epoch_step3: 
            if epoch % 100 == 0:
                with open(model_config.directory + 'log.txt', 'a') as log:
                    log.write(f'epoch: {epoch}\n')
                    # Check Disc
                    with torch.no_grad():
                        d_x_pu, d_x_u = model.check_disc(x_pl_batch, x_u_batch)
                        #d_x_pu, d_x_u = model.check_disc(fixed_pl, fixed_u)
                    log.write(f'd_x_pu: {d_x_pu.float().mean().item()}, d_x_u: {d_x_u.float().mean().item()}\n')     
                    # Check PN
                    with torch.no_grad():
                        d_x_pu2, d_x_pl2 = model.check_pn(x_pl_batch, x_u_batch)
                        #d_x_pu2, d_x_pl2 = model.check_pn(fixed_pl, fixed_u)
                    log.write(f'd_x_pu2: {d_x_pu2.float().mean().item()}, d_x_pl2: {d_x_pl2.float().mean().item()}\n')                
                
            if not (model_config.num_epoch_step1 < epoch <= model_config.num_epoch_step2):
                lstLoss1.append(sum(lst_1) / len(lst_1))
                lstLoss2.append(sum(lst_2) / len(lst_2))
                lstLoss3.append(sum(lst_3) / len(lst_3))
                lstLoss4.append(sum(lst_4) / len(lst_4))

        # --- TÍNH TOÁN LOSS & ACCURACY ---
        if epoch > model_config.num_epoch_step1: 
            if not (model_config.num_epoch_step_pn1 < epoch <= model_config.num_epoch_step_pn2):
                avg_l5 = sum(lst_5) / len(lst_5) if lst_5 else 0
                with open(model_config.directory + 'log_PN.txt', 'a') as log2:
                    log2.write(f'epoch: {epoch}, loss: {avg_l5}\n')

                # acc test
                with torch.no_grad():
                    val_acc, val_pr, val_re = model.accuracy(dl_val)
                lstAcc.append(val_acc) 

                with open(model_config.directory + 'log_PN.txt', 'a') as log2:
                    log2.write(f'...val: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}'.format(val_acc, val_pr, val_re) + '\n')
                print('...val: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}'.format(val_acc, val_pr, val_re))

                with torch.no_grad():
                    val_loss = model.loss_val(x_val[:20].to(device), x_val[20:].to(device))

                # 2. Lưu vào List
                lstVal.append(val_loss)
                print(val_loss)
                lstLoss5.append(avg_l5)

                # 3. Ghi file log
                

                # 4. IN RA MÀN HÌNH (CHỈ IN MỖI 10 EPOCH) 
                # dòng này nên để ra ngoài để nó in từ đầu luôn
                # chứ hiện tại nó đang bị dính câu lệnh if ở trên
                # if epoch % 10 == 0 or epoch == model_config.num_epoch:
                #     print(f'epoch: {epoch}, loss PN: {avg_l5:.4f}')
                #     # Dòng này tớ đã thêm lại đầy đủ acc, pr, re, loss:
                #     print(f'...val: acc: {val_acc:.4f}, pr: {val_pr:.4f}, re: {val_re:.4f}, loss: {val_loss:.4f}')

        # Save Checkpoint
        if epoch % model_config.save_epoch == 0:
            ckpt_path = os.path.join(model_config.directory, 'training_checkpoints', f'ckpt_{epoch}.pth')
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_en_state_dict': model.model_en.state_dict(),
                'model_de_state_dict': model.model_de.state_dict(),
                'model_disc_state_dict': model.model_disc.state_dict(),
                'model_pn_state_dict': model.model_pn.state_dict(),
                'opt_en': opt_en.state_dict(),
                'opt_de': opt_de.state_dict(),
                'opt_disc': opt_disc.state_dict(),
                'opt_pn': opt_pn.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        # Update Lists for Plotting
        # if lst_1: lstLoss1.append(sum(lst_1) / len(lst_1))
        # if lst_2: lstLoss2.append(sum(lst_2) / len(lst_2))
        # if lst_3: lstLoss3.append(sum(lst_3) / len(lst_3))
        # if lst_4: lstLoss4.append(sum(lst_4) / len(lst_4))

        # --- TÍNH TOÁN THỜI GIAN ---
        epoch_end_time = time.time()
        run_time = epoch_end_time - epoch_start_time 
        total_elapsed = epoch_end_time - total_start_time 
        total_str = str(datetime.timedelta(seconds=int(total_elapsed)))

        if epoch <= model_config.num_epoch_step3:
            if not (model_config.num_epoch_step1 < epoch <= model_config.num_epoch_step2):
                lsttime1.append(run_time)
        if epoch > model_config.num_epoch_step1:
            if not (model_config.num_epoch_step_pn1 < epoch <= model_config.num_epoch_step_pn2):
                lsttime2.append(run_time)

        # In thời gian mỗi 10 epoch
        if epoch % 10 == 0 or epoch == model_config.num_epoch:
            print(f'Exp: {num_exp} / Epoch: {epoch} || 1 Epoch: {run_time:.2f}s || Total Ran: {total_str}')

    # ================= END TRAINING =================
    
    # Plotting final results
    # 1. VAE Loss: Biểu thị lỗi tái tạo ảnh (Reconstruction Error)
    plotLoss(lstLoss1, 
             model_config.directory + 'loss_vae.png',
             title="VAE Reconstruction Loss",
             y_label="Reconstruction Error")

    # 2. Discriminator Loss: Loss của bộ phân biệt (thường là Binary Cross Entropy)
    plotLoss(lstLoss2, 
             model_config.directory + 'loss_disc.png',
             title="Discriminator Loss",
             y_label="Disc Loss")

    # 3. Generator Loss: Loss của bộ sinh (để đánh lừa Discriminator)
    plotLoss(lstLoss3, 
             model_config.directory + 'loss_gen.png',
             title="Generator Loss",
             y_label="Gen Loss")

    # 4. Classifier Loss: Loss phân loại (chính là phần PU Learning Risk)
    plotLoss(lstLoss4, 
             model_config.directory + 'loss_cl.png',
             title="Classifier Loss (PU Risk)",
             y_label="Risk Value")
    
    np.savez(model_config.directory + 'PU_loss_val_VAEPU', loss=lstVal)

    # Final Test
    with torch.no_grad():
        acc, precision, recall = model.accuracy(dl_test)

    with open(model_config.directory + 'log_PN.txt', 'a') as log2:
        log2.write(f'final test : acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}')
    print(f'FINAL TEST : acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}')

    if lsttime1:
        print(f'Average Time Phase 1: {sum(lsttime1)/len(lsttime1):.4f} sec/epoch')
    if lsttime2:
        print(f'Average Time Phase 2: {sum(lsttime2)/len(lsttime2):.4f} sec/epoch')

    # 5. PN Loss: Loss của riêng mạng Positive-Negative (thành phần cốt lõi của PU Learning)
    plotLoss(lstLoss5, 
             model_config.directory + 'loss_PN.png',
             title="PN Classifier Loss",
             y_label="PN Risk")

    # 6. Validation Accuracy: Độ chính xác kiểm thử
    plotLoss(lstAcc, 
             model_config.directory + 'val_accuracy.png',
             title="Validation Accuracy",
             y_label="Accuracy")

    return model