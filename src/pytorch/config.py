import torch
import torch.nn as nn
from types import SimpleNamespace

config = SimpleNamespace(
  # thư mục và dữ liệu
  directory = './result/MNIST_35_y/Exp1/',
  data='MNIST',

  # Tham số huấn luyện
  num_epoch=800,
  num_epoch_pn=100,
  num_epoch_pre=100,
  num_epoch_step1 = 400,
  num_epoch_step_pn1 = 500,
  num_epoch_step_pn2 = 600,
  num_epoch_step2 = 500,
  num_epoch_step3 = 700,

  batch_size_l=50,
  batch_size_u=100,
  batch_size_u_pn=100,
  batch_size_l_pn=50,
  batch_size_val=100,
  batch_size_test=100,

  # kiến trúc mạng
  n_input=784,
  n_hidden_vae_e=[500, 500],
  n_h_y=100,
  n_h_o=100,
  n_hidden_vae_d=[500, 500],
  n_o=2,
  n_hidden_nevae_e=[50],
  n_z=50,
  n_hidden_nevae_d=[50],
  n_hidden_disc=[256],
  n_hidden_cl=[],
  n_hidden_pn=[300, 300, 300, 300],

  # trọng số loss
  alpha_gen = 1.,
  alpha_disc = 1.,
  alpha_vae = 1.,
  alpha_cl = 1.,
  alpha_ne = 1.,
  alpha_mi = 1.,
  alpha_o = 1.,
  alpha_vade = 1.,
  alpha_gen2 = 1.,
  alpha_disc2 = 1.,

  # tham số pu learning
  pi_pl=100/10000,
  pi_pu=4900/10000,
  pi_u=9900/10000,
  pi_given=None,

  # optimizer
  lr_pu=3e-4,
  lr_disc=3e-4,
  lr_pn=3e-4,
  betas=(0.9, 0.999),
  beta1=0.9,
  beta2=0.999,

  # khác
  mode='near_o',
  k_gan=1,
  save_epoch=100,
  bool_pn_pre=False,
  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
  #log_interval=10,
)