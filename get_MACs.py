#from t2t_ablation_shortcuts import T2T_ViT
from REDCNN.networks import RED_CNN
import torch
from ptflops import get_model_complexity_info
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = RED_CNN()
net = net.to(device)
macs, params = get_model_complexity_info(net, (1, 64, 64), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))