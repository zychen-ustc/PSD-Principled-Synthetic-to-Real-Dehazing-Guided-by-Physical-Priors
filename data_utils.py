import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms

import os
import cv2
import numpy as np
from PIL import Image
from random import randrange
from pdb import set_trace as bp

from utils import *
from source_target_transforms import *

    
class ValData(data.Dataset):
    
    def __init__(self, data_dir):
        super().__init__()

        self.noisy_name_list = os.listdir(data_dir)
        self.noisy_data_dir = data_dir
        
    def __getitem__(self, index):
        name = self.noisy_name_list[index]
        
        noisy_path = os.path.join(self.noisy_data_dir, self.noisy_name_list[index])
        noisy_img = cv2.imread(noisy_path, cv2.IMREAD_UNCHANGED)
        noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
        noisy_img = np.float32(noisy_img / 255.)
            
        noisy_img = torch.from_numpy(np.ascontiguousarray(noisy_img)).permute(2, 0, 1).float()
        
        return noisy_img, name
    
    def __len__(self):
        return len(self.noisy_name_list)

    
    
class FinetuneData_with_augment_newdata(data.Dataset):

    def __init__(self, crop_size, data_dir, low_dir, noise_level_img, weight, rand_map, fixed_noise=True, attention=False):

        super().__init__()

        self.noisy_name_list = os.listdir(os.path.join(data_dir, 'semi_train_att_ppm_ca_bc_67'))
        self.gt_path_list = os.listdir(os.path.join(data_dir, 'high'))
        self.low_path_list = os.listdir(os.path.join(data_dir, 'low'))

        self.crop_size = crop_size
        self.noisy_data_dir = os.path.join(data_dir, 'semi_train_att_ppm_ca_bc_67')
        self.gt_data_dir = os.path.join(data_dir, 'high')
        self.low_data_dir = low_dir
        self.noise_level_img = noise_level_img
        self.fixed_noise = fixed_noise
        self.attention = attention
        self.weight = weight
        self.rand_map = rand_map
        
        if self.fixed_noise == True:
            self.noise_list = np.random.normal(0, self.noise_level_img/255., (len(self.gt_path_list), 256, 256, 3))
            
        self.transform = transforms.Compose([
            RandomRotationFromSequence([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip()]) 

    def random_pixel_shuffle(self, en_img):
        
        out = []
        for i in range(4):
            out.append(F.conv2d(en_img, self.weight[i], bias=None, stride=2, padding=0, dilation=1, groups=3)[None])
            
        shuffled = torch.cat(out).permute(1, 2, 3, 4, 0)
        imgs = []
        
        for i in range(4):
            imgs.append((shuffled * self.rand_map[i]).sum(4)[0])

        return imgs
    
    
    def __getitem__(self, index):

        crop_width, crop_height =  self.crop_size, self.crop_size
        
        name = self.noisy_name_list[index]
        gt_name = 'normal' + name[3:-9] + '.png'
        low_name = name[:-9] + '.png'
        
        #noisy_img = cv2.imread(os.path.join(self.noisy_data_dir, self.gt_path_list[index].split('.')[0] + 'fakeB.png'))
        noisy_img = Image.open(os.path.join(self.noisy_data_dir, name))
        gt_img = Image.open(os.path.join(self.gt_data_dir, gt_name))
        low_img = Image.open(os.path.join(self.low_data_dir, low_name))
        #print(os.path.join(self.noisy_data_dir, self.gt_path_list[index].split('.')[0] + 'fakeB.png'))
        
        width, height = noisy_img.size
        
        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        noisy_img = noisy_img.crop((x, y, x + crop_width, y + crop_height))
        gt_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        low_img = low_img.crop((x, y, x + crop_width, y + crop_height))
        
        noisy_img, gt_img, low_img = self.transform((noisy_img, gt_img, low_img))
        noisy_img = np.asarray(noisy_img) / 255.
        gt_img = np.asarray(gt_img) / 255.
        low_img = np.clip(np.asarray(low_img), 1/255, 1.)
        
        #noisy_img = np.float32(noisy_img / 255.)
        #gt_img = np.float32(gt_img / 255.)
        #low_img = np.clip(np.float32(low_img / 255.), 1 / 255, 1.)
        
        if self.fixed_noise:
            noise = self.noise_list[index]
        else:
            noise = np.random.normal(0, self.noise_level_img/255., noisy_img.shape)
        if self.attention:
            noise = noise * (((noisy_img / low_img).mean(0) + (noisy_img / low_img).max()) / ((noisy_img / low_img).mean() + (noisy_img / low_img).max()))
            noise = (noise / noise.std()) * (self.noise_level_img / 255.)
        noisy_noisy_img = noisy_img + noise
        #gt_img += np.random.normal(0, self.noise_level_img/255., gt_img.shape)
        
        noisy_img = torch.from_numpy(np.ascontiguousarray(noisy_img)).permute(2, 0, 1).float()
        gt_img = torch.from_numpy(np.ascontiguousarray(gt_img)).permute(2, 0, 1).float()
        noisy_noisy_img = torch.from_numpy(np.ascontiguousarray(noisy_noisy_img)).permute(2, 0, 1).float()
        low_img = torch.from_numpy(np.ascontiguousarray(low_img)).permute(2, 0, 1).float()
        
        noisy_img = torch.clamp(noisy_img, 0., 1.)
        gt_img = torch.clamp(gt_img, 0., 1.)
        noisy_noisy_img = torch.clamp(noisy_noisy_img, 0., 1.)
        low_img = torch.clamp(low_img, 0., 1.)
        #noisy_imgs = self.random_pixel_shuffle(noisy_img[None])
        #gt_imgs = self.random_pixel_shuffle(gt_img[None])

        return noisy_noisy_img, noisy_img, gt_img, low_img


    def __len__(self):
        return len(self.gt_path_list) 