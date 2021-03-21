import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from data_utils import TrainData, ValData, TestData, TestData2
from basemodel1 import GCANet
from utils import to_psnr, print_log, validation, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
import numpy as np
import os
import torchvision.transforms as tfs
from PIL import Image

epoch = 14

#test_data_dir = '/data/nnice1216/Dehazing/unlabeled/'
#test_data_dir = '/data/nnice1216/Dehazing/HSTS/real-world/'
test_data_dir = '/data/nnice1216/Dehazing/JPEGImages/'

output_dir = '/output/epoch{}/'.format(epoch)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = GCANet()
net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

net.load_state_dict(torch.load('haze_current9_physical'))


test_data_loader = DataLoader(TestData2(test_data_dir), batch_size=1, shuffle=False, num_workers=8)
unloader = tfs.ToPILImage()
net.eval()

def get_dark_channel(I, w):
    _, _, H, W = I.shape
    maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
    dc = maxpool(0 - I[:, :, :, :])

    return -dc

def get_atmosphere(I, dark_ch, p):
    B, _, H, W = dark_ch.shape
    num_pixel = int(p * H * W)
    flat_dc = dark_ch.resize(B, H * W)
    flat_I = I.resize(B, 3, H * W)
    index = torch.argsort(flat_dc, descending=True)[:, :num_pixel]
    A = torch.zeros((B, 3)).to('cuda')
    for i in range(B):
        A[i] = flat_I[i, :, index[i][torch.argsort(torch.max(flat_I[i][:, index[i]], 0)[0], descending=True)[0]]]

    return A[:, :, None, None]

output_dir = '/output/base_JEPG/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with torch.no_grad():
    for batch_id, val_data in enumerate(test_data_loader):
        #if batch_id > 150:
        #    break
        haze, haze_A, name = val_data
        #del(haze_A)
        print(batch_id, 'BEGIN!')

        B, _, H, W = haze.shape
        haze.to(device)
        haze_A.to(device)
        dc = get_dark_channel(haze, 15)
        A = get_atmosphere(haze, dc, 0.0001)
        #haze_A.to(device)
        _, pred, _, _, _ = net(haze, haze_A, True)
        
        ts = torch.squeeze(pred.clamp(0, 1).cpu())
        vutils.save_image(ts, output_dir + name[0].split('.')[0] + '_MyModel_{}.png'.format(batch_id))
        
        print(name[0].split('.')[0] + 'DONE!')