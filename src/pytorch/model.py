import torch
import torch.nn as nn
from types import SimpleNamespace
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt


class VAEEncoder(nn.Module):

  def __init__(self, config):
    super (VAEEncoder, self).__init__()
    self.config = config

    # nhánh xử lý encoder y
    self.vae_en_y = self._make_layers(
        input_dim=self.config.n_input,
        hidden_dims=self.config.n_hidden_vae_e
    )

    # các lớp linear tính mu và log_sigma_sq cho nhánh y
    # n_hidden_vae_e[-1] là số neuron của lớp ẩn cuối cùng
    self.vae_en_y_mu = nn.Linear(self.config.n_hidden_vae_e[-1], self.config.n_h_y)
    self.vae_en_y_lss = nn.Linear(self.config.n_hidden_vae_e[-1], self.config.n_h_y)

    # nhánh xử lý o (đầu vào là x (n_input) concat với o)
    self.vae_en_o = self._make_layers(
        input_dim = self.config.n_input + self.config.n_o,
        hidden_dims = self.config.n_hidden_vae_e
    )

    # các lớp Linear tính Lmu và log_sigma_sq cho nhánh o
    self.vae_en_o_mu = nn.Linear(self.config.n_hidden_vae_e[-1], self.config.n_h_o)
    self.vae_en_o_lss = nn.Linear(self.config.n_hidden_vae_e[-1], self.config.n_h_o)

  def _make_layers(self, input_dim, hidden_dims):
    """hàm helper để tạo các khối [dense->BN->leakyReLU]"""
    layers = []
    curr_dim = input_dim
    for h_dim in hidden_dims:
      layers.append(nn.Linear(curr_dim, h_dim)) # nn.Linear(in_features, out_features) thực hiện phép tính y = xW + b
      layers.append(nn.BatchNorm1d(h_dim, momentum=0.01, eps=1e-3)) # PyTorch yêu cầu truyền vào số lượng features (h_dim) cho lớp BatchNorm
      layers.append(nn.LeakyReLU(0.2))
      curr_dim = h_dim
    return nn.Sequential(*layers)

  def encode(self, x, o):
    # --- THÊM DÒNG NÀY ĐỂ AN TOÀN ---
    # Nếu x đang là (N, C, H, W) hoặc (N, H, W) -> Duỗi thành (N, n_input)
    if len(x.shape) > 2:
        x = x.view(x.size(0), -1) 
    # -------------------------------- 

    # 1. xử lý nhánh y
    hidden_y = self.vae_en_y(x)
    y_mu = self.vae_en_y_mu(hidden_y)
    y_lss = self.vae_en_y_lss(hidden_y)

    # 2. xử lý nhánh o: concat x và o theo chiều feature (dim=1)
    # x shape: (batch, n_input), o shape: (batch, n_o)
    x_o_combined = torch.cat([x, o], dim=1)
    hidden_o = self.vae_en_o(x_o_combined)

    o_mu = self.vae_en_o_mu(hidden_o)
    o_lss = self.vae_en_o_lss(hidden_o)

    return y_mu, y_lss, o_mu, o_lss

  """Trong PyTorch, forward là hàm mặc định khi gọi model(x, o)"""
  def forward(self, x, o):
    return self.encode(x, o)
  
  
#=========================================================================
class VAEDecoder(nn.Module):
  def __init__(self, config):
    super(VAEDecoder, self).__init__()
    self.config = config

    # 1. xây dựng danh sách các lớp ẩn
    layers = []
    # đầu vào là sự kết hợp của latent y và latent o
    curr_dim = self.config.n_h_y + self.config.n_h_o

    for h_dim in self.config.n_hidden_vae_d:
      layers.append(nn.Linear(curr_dim, h_dim))
      layers.append(nn.BatchNorm1d(h_dim))
      layers.append(nn.LeakyReLU(0.2))
      curr_dim = h_dim

    # 2. lớp đầu ra - reconstruction layer
    # kiểm tra điều kiện  dữ liệu
    #if 'lamda' in self.config.data:
    layers.append(nn.Linear(curr_dim, self.config.n_input))

    # gom tất cả vào Sequential
    self.vae_de = nn.Sequential(*layers)

    # phần code cho IIC: Có thể là một biến thể dữ liệu khác (ví dụ: ảnh màu hoặc dữ liệu liên tục)
    # if 'IIC' in self.config.data:
    #   self.vae_de_mu = nn.Sequential(
    #     nn.Linear(self.config.n_hidden_vae_d[-1], self.config.n_input),
    #     nn.ReLU()
    #   )


  def decode(self, h_y, h_o, use_sigmoid=False):
    # kết hợp hai vector ẩn theo chiều ngang (feature dimension: kích thước đặc trưng)
    z = torch.cat([h_y, h_o], dim=1)

    # đi qua mạng decode
    recon = self.vae_de(z)

    # Nếu muốn output dạng xác suất 0-1 cho ảnh trắng đen
    if use_sigmoid:
      recon = torch.sigmoid(recon)

    return recon


  def forward(self, h_y, h_o, use_sigmoid=False):
    return self.decode(h_y, h_o, use_sigmoid)

#=========================================================================
class Discriminator(nn.Module):
  def __init__(self, config):
    super(Discriminator, self).__init__()
    self.config = config

    layers = []
    curr_dim = self.config.n_input

    # vòng lặp tạo các lớp ẩn
    for h_dim in self.config.n_hidden_disc:
      layers.append(nn.Linear(curr_dim, h_dim))
      layers.append(nn.LeakyReLU(0.2))
      layers.append(nn.Dropout(0.3))
      curr_dim = h_dim

    # lớp output cuối cùng (Dense(1))
    layers.append(nn.Linear(curr_dim, 1))

    self.disc_u = nn.Sequential(*layers)

  def discriminate(self, x, use_sigmoid=False):
    disc = self.disc_u(x)
    if use_sigmoid:
      disc = torch.sigmoid(disc)
    return disc

  def forward(self, x, use_sigmoid=False):
    return self.discriminate(x, use_sigmoid)
#=========================================================================
class ClassifierO(nn.Module):
  def __init__(self, config):
    super(ClassifierO, self).__init__()
    self.config = config

    layers = []
    # đầu vào là kích thước n_h_o
    curr_dim = self.config.n_h_o

    # 1. vòng lặp các lớp ẩn
    for h_dim in self.config.n_hidden_cl:
      layers.append(nn.Linear(curr_dim, h_dim))
      layers.append(nn.BatchNorm1d(h_dim))
      layers.append(nn.LeakyReLU(0.2))
      curr_dim = h_dim

    #2. lớp đầu ra
    layers.append(nn.Linear(curr_dim, 1))

    self.classification = nn.Sequential(*layers)


  def classify(self, x, use_sigmoid=False):
    c = self.classification(x)

    if use_sigmoid:
      c = torch.sigmoid(c)

    return c

  def forward(self, x, use_sigmoid=False):
    return self.classify(x, use_sigmoid)

# Sau này khi train:
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
# weight_decay chính là L2 Regularization

#=========================================================================
class ClassifierPN(nn.Module):
    def __init__(self, config):
        super(ClassifierPN, self).__init__()
        self.config = config

        layers = []
        # Đầu vào là ảnh gốc (784 chiều)
        curr_dim = self.config.n_input

        # Vòng lặp tạo 4 lớp ẩn [300, 300, 300, 300]
        for h_dim in self.config.n_hidden_pn:
            # 1. Linear (Dense)
            layers.append(nn.Linear(curr_dim, h_dim))

            # 2. BatchNorm (Thay vì Dropout như Discriminator)
            layers.append(nn.BatchNorm1d(h_dim))

            # 3. Activation
            layers.append(nn.LeakyReLU(0.2))

            # Cập nhật chiều đầu vào cho lớp sau
            curr_dim = h_dim

        # Lớp đầu ra cuối cùng: Gom về 1 giá trị duy nhất
        layers.append(nn.Linear(curr_dim, 1))

        # Đóng gói
        self.classification = nn.Sequential(*layers)

    def classify(self, x, use_sigmoid=False):
        c = self.classification(x)

        if use_sigmoid:
            c = torch.sigmoid(c)

        return c

    # Hàm forward chuẩn của PyTorch
    def forward(self, x, use_sigmoid=False):
        return self.classify(x, use_sigmoid)


# Sau này khi train:
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
# weight_decay chính là L2 Regularization

#=========================================================================
class MyPU:
  def __init__(self, config,
               model_en, model_de, model_disc, model_cl, model_pn,
               opt_en, opt_de, opt_disc, opt_cl, opt_pn, opt_param=None):
    
    self.config = config

    # Lưu các mô hình
    self.model_en = model_en
    self.model_de = model_de
    self.model_disc = model_disc
    self.model_cl = model_cl
    self.model_pn = model_pn

    # Lưu trữ các bộ tối ưu hóa (The Optimizers)
    self.opt_en = opt_en
    self.opt_de = opt_de
    self.opt_disc = opt_disc
    self.opt_cl = opt_cl
    self.opt_pn = opt_pn

    # Dành cho phiên bản v2.0 nếu cần train tham số riêng
    self.opt_param = opt_param


  def reparameterization(self, mu, log_sig_sq):
    """
    Kỹ thuật Reparameterization Trick: z = mu + sigma * epsilon
    """
    std = torch.exp(0.5 * log_sig_sq)
    eps = torch.randn_like(std) # .randn_like(std) tự động lấy kích thước và device giống std
    return mu + std * eps



  def concat_datas(self, x_pl, x_u):
    """
    Gộp dữ liệu Positive (pl) và Unlabeled (u) lại.
    Đồng thời tạo nhãn one-hot 'o' tương ứng.
    """
    # 1. nối dữ liệu ảnh x theo chiều dọc (dim=0)
    x = torch.cat([x_pl, x_u], dim=0)

    # Quan trọng: Lấy device hiện tại của dữ liệu (CPU hoặc CUDA)
    # Để đảm bảo các tensor mới tạo ra cũng nằm cùng chỗ.
    device = x_pl.device

    # lấy số lượng mẫu
    n_pl = x_pl.size(0)
    n_u = x_u.size(0)

    # 2. Tạo vector 'o'
    # o_pl = [1, 0] cho dữ liệu Positive
    ones_pl = torch.ones(n_pl, 1, device=device)
    zeros_pl = torch.zeros(n_pl, 1, device=device)
    o_pl = torch.cat([ones_pl, zeros_pl], dim=1)

    # o_u = [0, 1] cho dữ liệu Unlabeled
    zeros_u = torch.zeros(n_u, 1, device=device)
    ones_u = torch.ones(n_u, 1, device=device)
    o_u = torch.cat([zeros_u, ones_u], dim=1)

    # Nối o lại
    o = torch.cat([o_pl, o_u], dim=0)

    return x, o


  def pretrain(self, x_pl, x_u):
    """
    Giai đoạn Pre-train: Chỉ huấn luyện Encoder và Decoder để tái tạo ảnh.
    Chưa đụng đến Discriminator hay Classifier.
    """

    # 1. chuẩn bị dữ liệu
    x, o = self.concat_datas(x_pl, x_u)

    # 2. xoá gradiant cũ (bắt buộc trong Pytorch)
    self.opt_en.zero_grad()
    self.opt_de.zero_grad()

    # forward pass- lan truyền xuôi
    # gọi encode để lấy tham số phân phối
    h_y_mu, h_y_log_sig_sq, h_o_mu, h_o_log_sig_sq = self.model_en.encode(x, o)

    # sampling
    h_y = self.reparameterization(h_y_mu, h_y_log_sig_sq)
    h_o = self.reparameterization(h_o_mu, h_o_log_sig_sq)

    # tái tạo ảnh - decode trả về logits haowjc mean tùy config
    recon_x = self.model_de.decode(h_y, h_o)

    dataset = self.config.data

    # 4. tính loss
    data_name = self.config.data 
    
    if any(name in data_name for name in ['MNIST', '01', 'Swiss', 'lamda']):
        # Dữ liệu nhị phân -> Dùng BCE Loss
        # x phải trong khoảng [0, 1]
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            recon_x, x, reduction='none'
        )
        # Sum over features, Mean over batch
        loss = torch.mean(torch.sum(bce_loss, dim=1))
        
    elif any(name in data_name for name in ['20', 'CIFAR']):
        # Dữ liệu liên tục -> Dùng MSE Loss (Gaussian Log-Likelihood)
        # Công thức gốc TF: 0.5 * (x - recon)^2
        mse = 0.5 * (x - recon_x) ** 2
        loss = torch.mean(torch.sum(mse, dim=1))
        
    else:
        # Mặc định hoặc báo lỗi
        raise NotImplementedError(f"Dataset {data_name} chưa được hỗ trợ hàm Loss")

    # 5. Backward Pass (lan truyền ngược)
    loss.backward()

    # 6. Cập nhật tham số
    self.opt_en.step()
    self.opt_de.step()

    # 7. trả về giá trị loss
    return loss.item()



  def findPrior(self, x_tr_l, x_tr_u):
    """
    Khởi tạo tham số Prior (p, mu, var) bằng thuật toán GMM.
    Sử dụng Sklearn vì Pytorch không có GMM tích hợp sẵn tốt như vậy.
    """

    # 1. Chuẩn bị dữ liệu One-hot (Tương tự hàm concat_datas)
    # Lưu ý: Chúng ta tạo thủ công ở đây để khớp logic với x_tr_l và x_tr_u riêng biệt
    device = x_tr_l.device

    # tạo o_pl
    n_l = x_tr_l.size(0)
    ones_l = torch.ones(n_l, 1, device=device)
    zeros_l = torch.zeros(n_l, 1, device=device)
    # kết hợp 2 cột
    o_l = torch.cat([ones_l, zeros_l], dim=1)

    # tạo o_u
    n_u = x_tr_u.size(0)
    ones_u = torch.ones(n_u, 1, device=device)
    zeros_u = torch.zeros(n_u, 1, device=device)
    # kết hợp 2 cột
    o_u = torch.cat([zeros_u, ones_u], dim=1)

    # 2. đưa qua encode để lấy biến tìm ẩn
    # Quan trọng: Dùng torch.no_grad() vì đây chỉ là bước khởi tạo, không train
    with torch.no_grad():
      # encode cho U
      h_y_u_mu, _, _, _ = self.model_en.encode(x_tr_u, o_u)
      h_y_u = h_y_u_mu

      # encode cho PL
      h_y_l_mu,  _, _, _ = self.model_en.encode(x_tr_l, o_l)
      h_y_l = h_y_l_mu

    # 3. Gộp dữ liệu và chuyển sang cpu-numpy để dùng sklearn
    h_y = torch.cat([h_y_u, h_y_l], dim=0).cpu().numpy()

    # dữ liệu labeled dùng để kiểm tra sau này:
    h_y_l_np = h_y_l.cpu().numpy()

    # 4. chạy thuật toán Gaussian (clustering)
    # covariance_type='diag' tương đương với việc giả định các chiều độc lập (giống VAE)
    gmm = GaussianMixture(n_components=2, covariance_type='diag')
    gmm.fit(h_y)

    # 5. Lấy các tham số từ GMM
    # weights_: tỷ lệ phần tử (pi)
    # means_: trung bình (mu)
    # covariances_: phương sai (var)

    # Mặc định ban đầu: Giả sử cụm 1 là Positive
    self.p = gmm.weights_[1]
    self.mu = gmm.means_.astype(np.float32)
    self.var = gmm.covariances_.astype(np.float32)

    # 6. Kiểm tra và Đảo lại (Cluster Alignment) - PHẦN QUAN TRỌNG NHẤT
    # Mục tiêu: Đảm bảo Index 1 luôn là phe Positive (Label=1)
    # Tính Log-Likelihood: Dữ liệu Labeled khớp với Cụm 0 (c0) hay Cụm 1 (c1) hơn?
    # Công thức: -0.5 * (x - mu)^2 / var - 0.5 * log(var)

    # cụm 0:
    term1_0 = (h_y_l_np - self.mu[0])**2 / (self.var[0] + 1e-9)
    term2_0 = np.log(self.var[0] + 1e-9)
    c0 = np.mean(-0.5 * term1_0 - 0.5 * term2_0, axis=1) # Trung bình theo feature

    # cụm 1
    term1_1 = (h_y_l_np - self.mu[1])**2 / (self.var[1] + 1e-9)
    term2_1 = np.log(self.var[1] + 1e-9)
    c1 = np.mean(-0.5 * term1_1 - 0.5 * term2_1, axis=1)

    # đếm xem có bao nhiêu mẫu PL thích cụm 0 hơn cụm 1
    num0 = np.sum(c0 > c1)
    frac0 = num0 / n_l

    # Nếu hơn 50% dữ liệu Labeled lại thích cụm 0
    # -> Nghĩa là Cụm 0 mới thực sự là Positive
    # -> Ta cần ĐẢO (Swap) vị trí để Index 1 luôn là Positive
    if frac0 > 0.5:
      self.p = gmm.weights_[0]
      # Đảo vị trí mu và var
      # Lưu ý: Cú pháp Python a, b = b, a rất tiện
      self.mu[0], self.mu[1] = self.mu[1].copy(), self.mu[0].copy()
      self.var[0], self.var[1] = self.var[1].copy(), self.var[0].copy()

    # 7. Chuyển ngược lại thành PyTorch Tensor và lưu vào Model
    # Chuyển thành Parameter hoặc Buffer để lưu trong state_dict nếu cần
    self.p = torch.tensor(self.p, device=device, dtype=torch.float32)
    self.mu = torch.tensor(self.mu, device=device, dtype=torch.float32)
    self.var = torch.tensor(self.var, device=device, dtype=torch.float32)

    # In ra để kiểm tra
    print(f'GMM Initialized: p={self.p:.4f}')
    # print("Mu shape:", self.mu.shape)



  def generate(self, x_pl, x_u, mode='near_o'):
    """
    Sinh dữ liệu mới bằng cách lai ghép đặc trưng giữa Positive và Unlabeled.

    Args:
        x_pl: Dữ liệu Positive (Labeled)
        x_u: Dữ liệu Unlabeled
        mode: 'near_o' (tìm láng giềng theo style), 'near_y' (tìm theo content), hoặc 'random'
    """
    device = x_pl.device
    
    # --- 1. AN TOÀN: Flatten dữ liệu đầu vào nếu là ảnh 3D/4D ---
    if len(x_pl.shape) > 2:
        x_pl = x_pl.view(x_pl.size(0), -1)
    if len(x_u.shape) > 2:
        x_u = x_u.view(x_u.size(0), -1)
    # -----------------------------------------------------------

    # 1. chuẩn bị nhãn one-hot
    n_pl = x_pl.size(0)
    n_u = x_u.size(0)

    o_pl = torch.cat([torch.ones(n_pl, 1, device=device),
                      torch.zeros(n_pl, 1, device=device)], dim=1)
    o_u = torch.cat([torch.zeros(n_u, 1, device=device),
                    torch.ones(n_u, 1, device=device)], dim=1)
    
    use_sigmoid = False
    if any(x in self.config.data for x in ['MNIST', '01', 'lamda']):
      use_sigmoid = True

    # =========================================================
    # MODE 1: NEAR_O (Tìm Unlabeled có STYLE gần nhất với Positive)
    # =========================================================
    if mode == 'near_o':
      # Encode Positive -> Lấy h_y (nội dung) và h_o (style)
      h_y_mu, h_y_log_sig_sq, h_o_mu, h_o_log_sig_sq = self.model_en.encode(x_pl, o_pl)
      h_y = self.reparameterization(h_y_mu, h_y_log_sig_sq)
      h_o = self.reparameterization(h_o_mu, h_o_log_sig_sq)

      # Encode Unlabeled -> Chỉ cần lấy h_o_x (style của Unlabeled)
      _, _, h_o_mu, h_o_log_sig_sq = self.model_en.encode(x_u, o_u)
      h_o_x = self.reparameterization(h_o_mu, h_o_log_sig_sq)

      # --- TÍNH KHOẢNG CÁCH (PRO VERSION) ---
      # Thay vì công thức a^2 - 2ab + b^2 dài dòng, PyTorch có hàm cdist cực xịn
      # Tính khoảng cách giữa từng vector h_o (của Pl) với TOÀN BỘ h_o_x (của U)
      # distance shape: [n_pl, n_u]
      distance = torch.cdist(h_o, h_o_x, p=2)

      # Tìm index của mẫu Unlabeled gần nhất cho mỗi mẫu Positive
      # argmin(dim=1) trả về index có giá trị nhỏ nhất trên mỗi dòng
      lstIdx = torch.argmin(distance, dim=1)

      # chọn ra các vector style h_o_x tương ứng
      ne_h_o = h_o_x[lstIdx] # lấy latent style
      x_u_select = x_u[lstIdx] # lấy ảnh gốc tương ứng (để debug)

      # GIẢI MÃ: Kết hợp h_y (của Positive) + ne_h_o (Style mượn của Unlabeled)
      # Logic config giữ nguyên như bản gốc
      x_pu = self.model_de.decode(h_y, ne_h_o, use_sigmoid=use_sigmoid)

      return h_y, h_o, ne_h_o, x_pu, x_u_select

    else:
      raise NotImplementedError(f'Mode {mode} not implemented')


  def sigmoid_loss(self, t, y):
    # Công thức gốc: tf.nn.sigmoid(-t * y)
    # PyTorch tương đương: torch.sigmoid(-t * y)
    return torch.sigmoid(-t * y)


  def train_step_vae(self, x_pl, x_u, epoch):
    # 1. chuẩn bị chế độ huấn luyện
    self.model_en.train()
    self.model_de.train()
    self.model_cl.train()
    # (Lưu ý: model_disc và model_pn ở đây dùng để tính loss,
    # nhưng ta không update weight của nó trong hàm này,
    # nên có thể eval hoặc train tùy logic toàn bài)
    self.model_disc.eval()
    self.model_pn.eval()

    device = x_pl.device
    alpha_gen = self.config.alpha_gen
    alpha_gen2 = self.config.alpha_gen2
    alpha_cl = self.config.alpha_cl

    # 2. Logic xác suất tiên nghiệm (Prior p)
    # p là xác suất một mẫu là Positive
    if getattr(self.config, 'pi_given', None) is None:
        p_val = self.config.pi_pl + self.config.pi_pu
    else:
        p_val = self.config.pi_given
    
    # [QUAN TRỌNG] Tạo tensor p trên đúng device để tránh lỗi runtime
    p = torch.tensor(p_val, device=device, dtype=torch.float32)

    # Đảm bảo p là tensor để tính toán
    # (Nếu self.p là tham số học được thì dùng self.p

    # 3. ghép dữ liệu
    x, o = self.concat_datas(x_pl, x_u)

    # =========================================================
    # BẮT ĐẦU TÍNH TOÁN (FORWARD PASS)
    # =========================================================

    # reset gradient
    self.opt_en.zero_grad()
    self.opt_de.zero_grad()
    self.opt_cl.zero_grad()

    # A. Encode
    h_y_mu, h_y_log_sig_sq, h_o_mu, h_o_log_sig_sq = self.model_en.encode(x, o)

    # B. Reparameterization Trick
    h_y = self.reparameterization(h_y_mu, h_y_log_sig_sq)
    h_o = self.reparameterization(h_o_mu, h_o_log_sig_sq)

    # C. Tính toán xác suất thuộc cụm (Cluster Probability) - GMM Logic
    # Công thức log-likelihood của Gaussian: -0.5 * (x-mu)^2 / var - 0.5 * log(var)

    # Cụm 0 (Negative)
    c0 = (
    -0.5 * ((h_y - self.mu[0])**2) / (self.var[0])
    - 0.5 * torch.log(torch.clamp(self.var[0], min=1e-9))
    + torch.log(torch.clamp(torch.tensor(1 - p), min=1e-9))
    )

    # Cụm 1 (Positive)
    c1 = (
    - 0.5 * ((h_y - self.mu[1])**2) / (self.var[1])
    - 0.5 * torch.log(torch.clamp(self.var[1], min=1e-9))
    + torch.log(torch.clamp(torch.tensor(p), min=1e-9))
    )

    # Sum over features (axis=1)
    c0 = torch.sum(c0, dim=1, keepdim=True)
    c1 = torch.sum(c1, dim=1, keepdim=True)

    # Softmax để ra xác suất thuộc cụm (Posterior q(y|x))
    c_concat = torch.cat([c0, c1], dim=1)

    # Lấy xác suất của class 1 (Positive)
    c = torch.softmax(c_concat, dim=1)[:, 1].unsqueeze(1)

    # D. GAN Loss (Generator part)
    # Sinh mẫu giả (x_pu)
    # Lưu ý: generate trong logic cũ trả về tuple, ta lấy phần tử thứ 4 là x_pu
    _, _, _, x_pu, _ = self.generate(x_pl, x_u, mode=self.config.mode)

    # Discriminator (chú ý: sigmoid=False vì dùng BCEWithLogitsLoss)
    d_x_pu = self.model_disc.discriminate(x_pu, use_sigmoid=False)
    label_real = torch.ones_like(d_x_pu)

    # Loss: Generator muốn lừa Discriminator (d_x_pu -> 1)
    gan_loss = alpha_gen * torch.nn.functional.binary_cross_entropy_with_logits(d_x_pu, label_real)

    # E. GAN Loss 2 (Positive/Negative Classifier)
    if epoch <= self.config.num_epoch_step1:
      gan_loss2 = torch.tensor(0.0, device=device)
    else:
      d_x_pu2 = self.model_pn.classify(x_pu, use_sigmoid=False)
      gan_loss2 = alpha_gen2 * torch.mean(self.sigmoid_loss(d_x_pu2, label_real))

    # =========================================================
    # TÍNH TOÁN CÁC THÀNH PHẦN VADE LOSS (LOSS 1 -> 7)
    # =========================================================

    # Loss 1: KL Divergence cho h_y (Mixture of Gaussians)
    # Công thức: log(var_prior) + (var_post + (mu_post - mu_prior)^2) / var_prior
    # Chú ý: trong code gốc dùng exp(log_sig_sq) chính là variance
    term1_0 = torch.log(torch.clamp(self.var[0], min=1e-9)) + (torch.exp(h_y_log_sig_sq) + (h_y_mu - self.mu[0])**2) / self.var[0]
    loss1_0 = -0.5 * torch.sum(term1_0, dim=1, keepdim=True)

    term1_1 = torch.log(torch.clamp(self.var[1], min=1e-9)) + (torch.exp(h_y_log_sig_sq) + (h_y_mu - self.mu[1])**2) / self.var[1]
    loss1_1 = -0.5 * torch.sum(term1_1, dim=1, keepdim=True)

    # Kết hợp dựa trên xác suất c
    loss1 = torch.mean((1-c)*loss1_0 + c * loss1_1)

    # Loss 2: KL Divergence cho h_o (Standard Normal Prior)
    # Prior là N(0, 1) -> công thức đơn giản hơn
    loss2 = -torch.mean(torch.sum(0.5 * (torch.exp(h_o_log_sig_sq) + h_o_mu**2), dim=1))

    # loss 3:Reconstruction(tái cấu trúc) Loss (Quan trọng cho lamda)
    recon_x = self.model_de.decode(h_y, h_o) # logits (chưa có sigmoid)

    if any(x in self.config.data for x in ['MNIST', '01', 'lamda']):
      # Binary Cross Entropy (cho ảnh đen trắng/binary)
      bce = torch.nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='none')
      loss3 = -torch.mean(torch.sum(bce, dim=1))

    elif 'conv' in self.config.data:
      # Convolution output 3 chiều [C, H, W]
      bce = torch.nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='none')
      loss3 = -torch.mean(torch.sum(bce, dim=[1, 2, 3]))

    else: # Continuous data (CIFAR, 20) -> MSE Loss
      mse = 0.5 * (x - torch.sigmoid(recon_x))**2 # Hoặc recon_x nếu decode đã có sigmoid?
      # Code TF '20': reduce_sum(0.5 * square(x - recon_x))
      # Giả sử decode trả về linear, x là thực.
      # Lưu ý: Code TF dùng square(x-recon_x). Nếu recon_x là logits thì sai.
      # Nhưng tôi sẽ giữ logic: decode trả về Linear, loss3 dùng MSE.
      loss3 = self.config.alpha_test * -torch.mean(torch.sum(0.5 * (x - recon_x)**2, dim=1))


    # Loss 4 & 5: Các thành phần Entropy (Regularization)
    # Giữ nguyên logic TF: 0.5 * (1 + log_var)
    loss4 = torch.mean(torch.sum(0.5 * (1 + h_y_log_sig_sq), dim=1))
    loss5 = torch.mean(torch.sum(0.5 * (1 + h_o_log_sig_sq), dim=1))

    # Loss 6: Entropy của phân phối cụm q(y|x)
    # Logic: -c*log(c/p) - (1-c)*log((1-c)/(1-p))
    # Thêm epsilon 1e-9 để tránh log(0)
    term6_1 = c * torch.log((c / torch.clamp(torch.tensor(p), min=1e-9)) + 1e-9)
    term6_2 = (1 - c) * torch.log(((1 - c) / torch.clamp(torch.tensor(1 - p), min=1e-9)) + 1e-9)

    loss6 = torch.mean(- term6_1 - term6_2)

    # Loss 7: Observation Classifier Loss
    # Phân loại xem h_o đến từ nguồn nào (o)
    c_o = self.model_cl(h_o, use_sigmoid=False) # logits
    # Lấy cột đầu tiên: 1 nếu là PL, 0 nếu là U
    label_o = o[:, 0].unsqueeze(1)
    loss7 = -torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(c_o, label_o))

    # TỔNG HỢP LOSS
    # vade_loss = -alpha * (sum all components)
    # Vì ở trên ta dùng dấu trừ (-) cho các thành phần loss (maximize ELBO),
    # nên vade_loss sẽ là minimization.
    vade_loss = -self.config.alpha_vade * (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7)

    loss_t1 = vade_loss + gan_loss + gan_loss2

    # =========================================================
    # BACKWARD & UPDATE (OPTIMIZATION)
    # =========================================================
    loss_t1.backward()

    self.opt_en.step()
    self.opt_de.step()
    self.opt_cl.step()

    # Update params (pi, mu, var) nếu cần (đã comment trong code gốc)
    # if training_params:
    #    self.opt_param.step()

    return vade_loss.item(), gan_loss.item(), gan_loss2.item()


  def train_step_disc(self, x_pl, x_u):
    # 1. Bật chế độ train cho Discriminator
    self.model_disc.train()

    alpha_disc = self.config.alpha_disc
    alpha_disc2 = self.config.alpha_disc2 # (Chưa dùng tới trong đoạn này)

    # 2. Xóa gradient cũ
    self.opt_disc.zero_grad()

    # 3. Sinh ảnh (tương đương dòng generate trong TF)
    # Dùng no_grad và detach để giả lập hành vi "chỉ quan tâm disc_vars" của TF
    _, _, _, x_pu, _ = self.generate(x_pl, x_u, self.config.mode)

    x_pu = x_pu.detach()

    d_x_pu = self.model_disc.discriminate(x_pu, use_sigmoid=False)
    d_x_u = self.model_disc.discriminate(x_u, use_sigmoid=False)

    label_pu = torch.zeros_like(d_x_pu)
    label_u = torch.ones_like(d_x_u)

    loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(d_x_pu, label_pu)
    loss_real = torch.nn.functional.binary_cross_entropy_with_logits(d_x_u, label_u)

    disc_loss = alpha_disc * (loss_fake + loss_real)

    disc_loss.backward()

    self.opt_disc.step()

    return disc_loss.item()


  # def sigmoid_loss(self, t, y):
  #   # Công thức gốc: tf.nn.sigmoid(-t * y)
  #   # PyTorch tương đương: torch.sigmoid(-t * y)
  #   return torch.sigmoid(-t * y)

  def logistic_loss(self, t, y):
    return torch.nn.functional.softplus(-t * y)

  def ce_loss(self, t, y):
    return torch.nn.functional.binary_cross_entropy_with_logits(t, y, reduction='none')


  def train_step_pn_pre(self, x_pl, x_u):
    # 1. Bật chế độ Train cho model_pn (quan trọng cho Dropout/BatchNorm)
    self.model_pn.train()

    pi_pl = self.config.pi_pl
    pi_pu = self.config.pi_pu
    pi_u = self.config.pi_u

    pi_p = pi_pl + pi_pu

    # 3. Xóa gradient cũ trước khi tính mới
    self.opt_pn.zero_grad()

    pn_x_pl = self.model_pn.classify(x_pl, use_sigmoid=False)
    pn_x_u = self.model_pn.classify(x_u, use_sigmoid=False)

    # Loss cho tập PositiveLabeled
    pu1_loss = torch.mean(self.sigmoid_loss(pn_x_pl, torch.ones_like(pn_x_pl)))
    # bù Loss ước lượng cho tập Positive (Risk Positive)
    pu2_loss = -pi_p * torch.mean(self.sigmoid_loss(pn_x_pl, -torch.ones_like(pn_x_pl)))
    # Loss cho tập Unlabeled (Risk Unlabeled)
    u_loss = pi_u * torch.mean(self.sigmoid_loss(pn_x_u, -torch.ones_like(pn_x_u)))

    # Nếu phần (Risk_Positive + Risk_Unlabeled) < 0 -> Overfitting -> Phạt ngược lại
    risk_u = pu2_loss + u_loss
    if risk_u >= 0:
        pn_loss = pu1_loss + risk_u
    else:
        # Code gốc: pn_loss = - (pu2_loss + u_loss)
        # Logic: Tìm cách đẩy risk_u về 0
        pn_loss = -risk_u

    pn_loss.backward()
    self.opt_pn.step()

    return pu1_loss.item()
  

  def train_step_pn(self, x_pl, x_u):

    self.model_pn.train()

    self.opt_pn.zero_grad()

    pi_pl = self.config.pi_pl
    pi_pu = self.config.pi_pu
    pi_u = self.config.pi_u

    with torch.no_grad():
      _, _, _, x_pu, _ = self.generate(x_pl, x_u, self.config.mode)
      
    # Cắt đứt gradient hoàn toàn (để x_pu trở thành input tĩnh như TF)
    x_pu = x_pu.detach()

    pn_x_pl = self.model_pn.classify(x_pl, use_sigmoid=False)
    pn_x_u = self.model_pn.classify(x_u, use_sigmoid=False)
    pn_x_pu = self.model_pn.classify(x_pu, use_sigmoid=False)

    pl_loss = torch.mean(pi_pl * self.sigmoid_loss(pn_x_pl, torch.ones_like(pn_x_pl)))
    pu1_loss = torch.mean(pi_pu * self.sigmoid_loss(pn_x_pu, torch.ones_like(pn_x_pu)))
    pu2_loss = torch.mean(-pi_pu * self.sigmoid_loss(pn_x_pu, -torch.ones_like(pn_x_pu)))
    u_loss = torch.mean(pi_u * self.sigmoid_loss(pn_x_u, -torch.ones_like(pn_x_u)))

    risk_u = pu2_loss + u_loss
    if risk_u >= 0:
      pn_loss = pl_loss + pu1_loss + pu2_loss + u_loss
    else:
      pn_loss = - risk_u
    
    pn_loss.backward()
    self.opt_pn.step()

    return (pl_loss + pu1_loss + pu2_loss + u_loss).item()
  

  def compare_MIST(self, fname, x_pl, x_u, n=1):
        # 1. Chuyển Model sang chế độ "ngắm" (Evaluation Mode)
        # Để tắt Dropout, BatchNorm (giống no_grad, nhưng dành cho layer)
        self.model_en.eval()
        self.model_de.eval()
        self.model_pn.eval()

        # 2. Tắt bộ nhớ đạo hàm (Vì chỉ vẽ thôi, không học)
        with torch.no_grad():
            lst_x_pu = []
            plt.figure(figsize=(10, 10))

            # Chạy thử 3 lần để sinh ra 3 bộ ảnh giả lập khác nhau
            for _ in range(3):
                h_y, h_o, ne_h_o, x_pu, _ = self.generate(x_pl, x_u, self.config.mode)
                lst_x_pu.append(x_pu)
            
            # Tái tạo lại ảnh gốc (x_pl) xem model nén-giải nén có tốt không
            # Lưu ý: h_y, h_o ở đây lấy từ vòng lặp cuối cùng (giống code gốc)
            # Nếu hàm decode của cậu trả về logits, cậu cần bọc torch.sigmoid() bên ngoài
            # Tùy hàm decode của cậu viết thế nào, giả sử có tham số sigmoid=True
            recon_x_pl = self.model_de.decode(h_y, h_o, use_sigmoid=True)

            # --- BẮT ĐẦU VẼ ---
            
            # Hàm phụ trợ để convert Tensor PyTorch sang ảnh Numpy chuẩn
            def to_numpy_img(tensor, idx):
                # Lấy mẫu thứ idx, chuyển về CPU, cắt đạo hàm, chuyển sang numpy
                img = tensor[idx].cpu().detach().numpy()
                return img

            # Logic vẽ cho MNIST (Ảnh xám 28x28)
            if 'MNIST' in self.config.data:
                for i in range(1, n + 1):
                    # Dòng 1: Ảnh thật (Real)
                    plt.subplot(n, 5, (i - 1) * 5 + 1)
                    # Reshape về 28x28 để vẽ
                    plt.imshow(to_numpy_img(x_pl, i - 1).reshape(28, 28), vmin=0, vmax=1, cmap="gray")
                    plt.title("real")

                    # Dòng 2: Ảnh tái tạo (Reconstructed)
                    plt.subplot(n, 5, (i - 1) * 5 + 2)
                    plt.imshow(to_numpy_img(recon_x_pl, i - 1).reshape(28, 28), vmin=0, vmax=1, cmap="gray")
                    plt.title("fake_pl")

                    # Dòng 3,4,5: Ảnh sinh ra (Generated) từ 3 lần chạy
                    plt.subplot(n, 5, (i - 1) * 5 + 3)
                    plt.imshow(to_numpy_img(lst_x_pu[0], i - 1).reshape(28, 28), vmin=0, vmax=1, cmap="gray")
                    plt.title("fake_pu1")

                    plt.subplot(n, 5, (i - 1) * 5 + 4)
                    plt.imshow(to_numpy_img(lst_x_pu[1], i - 1).reshape(28, 28), vmin=0, vmax=1, cmap="gray")
                    plt.title("fake_pu2")

                    plt.subplot(n, 5, (i - 1) * 5 + 5)
                    plt.imshow(to_numpy_img(lst_x_pu[2], i - 1).reshape(28, 28), vmin=0, vmax=1, cmap="gray")
                    plt.title("fake_pu3")

            # Logic vẽ cho các dataset khác (CIFAR, v.v...)
            else:
                for i in range(1, n + 1):
                    # Hàm phụ xử lý ảnh màu (3 kênh)
                    def process_color_img(img_tensor):
                        img = to_numpy_img(img_tensor, i - 1)
                        # Nếu ảnh dạng (Channels, Height, Width) -> Đổi thành (Height, Width, Channels)
                        if img.shape[0] == 3 or img.shape[0] == 1: 
                            img = img.transpose(1, 2, 0)
                        return img

                    plt.subplot(n, 5, (i - 1) * 5 + 1)
                    plt.imshow(process_color_img(x_pl), vmin=0, vmax=1, cmap="gray")
                    plt.title("real")

                    plt.subplot(n, 5, (i - 1) * 5 + 2)
                    plt.imshow(process_color_img(recon_x_pl), vmin=0, vmax=1, cmap="gray")
                    plt.title("fake_pl")

                    plt.subplot(n, 5, (i - 1) * 5 + 3)
                    plt.imshow(process_color_img(lst_x_pu[0]), vmin=0, vmax=1, cmap="gray")
                    plt.title("fake_pu1")

                    plt.subplot(n, 5, (i - 1) * 5 + 4)
                    plt.imshow(process_color_img(lst_x_pu[1]), vmin=0, vmax=1, cmap="gray")
                    plt.title("fake_pu2")

                    plt.subplot(n, 5, (i - 1) * 5 + 5)
                    plt.imshow(process_color_img(lst_x_pu[2]), vmin=0, vmax=1, cmap="gray")
                    plt.title("fake_pu3")

            plt.savefig(fname)
            plt.close()
            
            # Quan trọng: Trả lại chế độ Train sau khi vẽ xong
            self.model_en.train()
            self.model_de.train()
            self.model_pn.train()


# vẽ biểu đồ cho dữ liệu bảng
  def compare(self, fname, x_pl, x_u, n=1):
        # 1. Chế độ ngắm
        self.model_en.eval()
        self.model_de.eval()
        self.model_pn.eval()

        with torch.no_grad():
            lst_x_pu = []
            # Sinh dữ liệu giả
            for _ in range(3):
                h_y, h_o, ne_h_o, x_pu, _ = self.generate(x_pl, x_u, self.config.mode)
                lst_x_pu.append(x_pu)

            # Tái tạo dữ liệu thật
            recon_x_pl = self.model_de.decode(h_y, h_o, use_sigmoid=True) # Hoặc False tùy activation cuối của cậu
            
            # --- VẼ BIỂU ĐỒ SO SÁNH ---
            # Thay vì vẽ ảnh, ta vẽ biểu đồ so sánh giá trị các features
            plt.figure(figsize=(15, 4 * n)) # Chỉnh kích thước ảnh

            for i in range(n): # Duyệt qua n mẫu đầu tiên
                
                # Lấy dữ liệu ra CPU và Numpy
                real_data = x_pl[i].cpu().detach().numpy()
                recon_data = recon_x_pl[i].cpu().detach().numpy()
                fake_data = lst_x_pu[0][i].cpu().detach().numpy() # Lấy đại diện 1 cái fake để so

                num_features = len(real_data)
                indices = range(num_features)

                # Tạo subplot: Mỗi mẫu 1 hàng
                plt.subplot(n, 1, i + 1)
                
                # Vẽ 3 đường hoặc 3 loại cột để so sánh
                plt.plot(indices, real_data, 'b-', label='Real', marker='o', alpha=0.7)
                plt.plot(indices, recon_data, 'g--', label='Recon', linestyle='dashed')
                plt.plot(indices, fake_data, 'r:', label='Generated', alpha=0.7)
                
                plt.title(f"Sample {i+1}: Comparison of Features")
                plt.xlabel("Feature Index")
                plt.ylabel("Value")
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()

        # Trả lại chế độ train
        self.model_en.train()
        self.model_de.train()
        self.model_pn.train()


  def check_disc(self, x_pl, x_u):
    # 1. Chuyển sang chế độ đánh giá (Evaluation Mode)
    # Rất quan trọng nếu model có Dropout hoặc BatchNorm
    self.model_disc.eval()
    self.model_en.eval() # Vì hàm generate dùng model_en/de
    self.model_de.eval()

    # 2. Tắt bộ máy tính đạo hàm (Tiết kiệm bộ nhớ)
    with torch.no_grad():
      _, _, _, x_pu,_ = self.generate(x_pl, x_u, self.config.mode)
      d_x_pu = self.model_disc.discriminate(x_pu, use_sigmoid=True)
      d_x_u = self.model_disc. discriminate(x_u, use_sigmoid=True)

    # 3. Trả lại chế độ Train cho các model (Để không ảnh hưởng các hàm sau)
    # Bước này cực quan trọng, nếu quên thì model sẽ không học được nữa!
    self.model_disc.train()
    self.model_en.train()
    self.model_de.train()

    return d_x_pu, d_x_u
  
  def check_pn(self, x_pl, x_u):
    # 1. chuyển về chế độ đánh giá:
    self.model_pn.eval()

    # 2. tắt tính năng đạo hàm, tiết kiệm bộ nhớ
    with torch.no_grad():
      _, _, _, x_pu,_ = self.generate(x_pl, x_u, self.config.mode)
      d_x_pu = self.model_pn.classify(x_pu, use_sigmoid=True)
      d_x_pl = self.model_pn.classify(x_pl, use_sigmoid=True)
    
    # 3. trả về chế độ train cho các model 
    self.model_pn.train()
    self.model_en.train()
    self.model_de.train()

    return d_x_pu, d_x_pl


  def check_cl(self, x_pl, x_u):
    self.model_cl.eval()
    self.model_en.eval()
    self.model_de.eval()

    idx_pl = torch.randperm(x_pl.size(0))[:10]
    idx_u = torch.randperm(x_u.size(0))[:10]

    # Cắt dữ liệu theo index (Giữ nguyên Tensor trên GPU)
    x_pl_batch = x_pl[idx_pl]
    x_u_batch = x_u[idx_u]

    with torch.no_grad():
      _, h_o, ne_h_o, _, _ = self.generate(x_pl_batch, x_u_batch, self.config.mode) 

      c_h_o = self.model_cl.classify(h_o, use_sigmoid=True)
      c_ne_h_o = self.model_cl.classify(ne_h_o, use_sigmoid=True)

      _, _, h_o_mu_2, h_o_log_sig_sq_2 = self.model_en.encode(x_u_batch)
      h_o_2 = self.reparameterization(h_o_mu_2, h_o_log_sig_sq_2)

      c_h_o_2 = self.model_cl.classify(h_o_2, use_sigmoid=True)

    self.model_cl.train()
    self.model_disc.train()
    self.model_en.train()
    self.model_de.train()
    
    return c_h_o, c_ne_h_o, c_h_o_2
  

  def accuracy(self, dataset):
    # 1. chuyển về chế độ đi thi
    self.model_pn.eval()

    tp =0
    tn = 0
    fp = 0
    fn = 0

    # 2. tắt đạo hàm để tiết kiệm bộ nhớ nha
    with torch.no_grad():
      for x_val, y_val in dataset:
        # chuyển dữ liệu lên gpu
        # Ta lấy device từ parameters của model cho chắc ăn
        device = next(self.model_pn.parameters()).device
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        # dự đoán ( lấy logits, sigmoid=False là lấy giá trị thô chưa ép về 0-1)
        c = self.model_pn.classify(x_val, use_sigmoid=False)

        # làm phẳng tensor để dễ so sánh - thành vector 1 chiều
        # Ví dụ: từ [64, 1] -> [64]
        c = c.view(-1)
        y_val = y_val.view(-1)

        # --- TÍNH TOÁN (Logic chuyển từ TF sang PyTorch) ---
        # Bước A: Lọc ra các dự đoán tương ứng với nhãn thật
        # (Tương đương tf.boolean_mask)
        preds_of_positives = c[y_val == 1] # những mẫu có nhãn là 1
        preds_of_negative = c[y_val == -1] # những mẫu có nhãn là -1 

        # Bước B: So sánh với ngưỡng 0 (Vì dùng Logits)

        # TP: nhãn là 1 dự đoán > 0
        tp += (preds_of_positives > 0).sum().item()

        # FN: nhãn là 1, dự đoán <=0 (bị sót)
        fn += (preds_of_positives <= 0).sum().item()

        # FP: nhãn là -1 , dự đoán > 0 ( báo động giả)
        fp += (preds_of_negative > 0).sum().item()

        # TN: nhãn -1, dự đoán <= 0 (đoán đúng là âm)
        tn += (preds_of_negative <= 0).sum().item()

    # 3. tính toán các chỉ số
    if tp + fp == 0: # chống lỗi chia cho 0 (Zero Division Error).
      precision = 0.0
    else:
      precision = tp / (tp + fp) 
    
    if tp + fn == 0: # chống lỗi chia cho 0 (Zero Division Error).
      recall = 0.0
    else:
      recall = tp / (tp + fn)
    
    if (tp + tn + fp + fn) == 0:
      acc = 0.0
    else:
      acc = (tp + tn) / (tp + tn + fp +fn)

    # 4. trả lại chế đọ train
    self.model_pn.train()

    return acc, precision, recall


  def loss_val(self, x_pl, x_u):
    self.model_pn.eval()
    self.model_en.eval() # generate có dùng
    self.model_de.eval()
    self.model_disc.eval()

    pi_pl = self.config.pi_pl
    pi_pu = self.config.pi_pu
    pi_u = self.config.pi_u

    pi_p = pi_pl + pi_pu

    with torch.no_grad():
      _, _, _, x_pu, _ = self.generate(x_pl, x_u, self.config.mode)

      pn_x_pl = self.model_pn.classify(x_pl, use_sigmoid=False)
      pn_x_pu = self.model_pn.classify(x_pu, use_sigmoid=False)
      pn_x_u = self.model_pn.classify(x_u, use_sigmoid=False)

      pl_loss = torch.mean(pi_pl * self.sigmoid_loss(pn_x_pl, torch.ones_like(pn_x_pl)))
      pu1_loss = torch.mean(pi_pu * self.sigmoid_loss(pn_x_pu, torch.ones_like(pn_x_pu)))
      pu2_loss = torch.mean(-pi_pu * self.sigmoid_loss(pn_x_pu, -torch.ones_like(pn_x_pu)))
      u_loss = torch.mean(pi_u * self.sigmoid_loss(pn_x_u, -torch.ones_like(pn_x_u)))

      total_loss = pl_loss + pu1_loss + pu2_loss + u_loss

    self.model_pn.train()
    self.model_en.train()
    self.model_de.train()
    self.model_disc.train()

    return total_loss.item()