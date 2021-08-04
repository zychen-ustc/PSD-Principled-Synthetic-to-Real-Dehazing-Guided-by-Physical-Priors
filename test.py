import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as tfs

import os
import cv2
import random
from PIL import Image
from pdb import set_trace as bp

from PDNet import DnCNN_c, Estimation_direct, DnCNN_finetune, DecomNet
from data_utils import *

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


val_data_dir = './test_imgs/'
val_loader = ValData(val_data_dir)
val_data_loader = DataLoader(val_loader)

n_channels = 3
nb = 20
do_ps = False
pss = 2

c = 3
model = DnCNN_c(channels=c, num_of_layers=20, num_of_est = 2 * c)
model.to(device)
model = nn.DataParallel(model, device_ids=device_ids)
model.load_state_dict(torch.load('./checkpoints/dncnn_net.pth'))
model.eval()

est_net = Estimation_direct(c, 2 * c)
est_net.to(device)
est_net = nn.DataParallel(est_net, device_ids=device_ids)
est_net.load_state_dict(torch.load('./checkpoints/est_net.pth'))
est_net.eval()

decom_net = DecomNet()
decom_net.to('cuda')
decom_net = nn.DataParallel(decom_net, device_ids=device_ids)
decom_net.load_state_dict(torch.load('./checkpoints/decom_net.pth'))
decom_net.eval()

crop_size = 60
batch_size = 1
do_ps = False
scale = 2
weight = 0
rand_map = 0


final_dir = './results/'
if not os.path.exists(final_dir):
    os.makedirs(final_dir)
    
toimg = tfs.ToPILImage()

miu_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


iter_num = len(miu_list)
with torch.no_grad():
    for batch_id, data in enumerate(val_data_loader):
        print(len(val_data_loader))
        print('batch{} begin'.format(batch_id))

        noisy_img, name = data

        y = noisy_img.to(device)

        z = y.clone()
        
        est0 = torch.clamp(est_net(y), 0., 1.).clone().detach()
        noise_level = est0[:, :3].mean().item() * 100

        for i in range(iter_num):
            
            miu = miu_list[i] + 0.5
            
            if i == 0:
                est = torch.clamp(est_net(z), 0., 1.)
                res = model(z, est)
                x = torch.clamp(z - res, 0., 1.)
            
            else:
                z_temp = torch.clamp((z), 0., 1.)
                # z_temp = torch.clamp((z + z_before) / 2, 0., 1.)
                est = torch.clamp(est_net(z_temp), 0., 1.)
                res = model(z_temp, est)
                x = torch.clamp(z_temp - res, 0., 1.)
                
                if i == iter_num - 1:
                    for ind in range(x.shape[0]):
                        toimg(x[ind].cpu()).save(final_dir + '{}.png'.format(name[ind], i))
                 
            est = torch.clamp(est_net(x), 0., 1.).clone().detach()
            noise_level = (est[0, :3].mean() * 100).item()
            noise_map = torch.normal(0, noise_level/255, noisy_img.shape).type(torch.FloatTensor).to(device)
            est_3 = torch.cat([est[:, :3].mean(1)[:, None], est[:, :3].mean(1)[:, None], est[:, :3].mean(1)[:, None]], dim=1)
            noise_map = noise_map * (1 + est_3)
            
            nx = torch.clamp(x + noise_map, 0., 1.)
            # z_before = z
            z = torch.clamp((miu * nx + y) / (1 + miu), 0., 1.)