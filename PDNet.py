'''
Minor Modification from  https://github.com/SaoYan/DnCNN-PyTorch SaoYan
Re-implemented by Yuqian Zhou
'''
import torch
import torch.nn as nn
from models.network_dncnn import DnCNN as DnCNN2
import torch.nn.functional as F

class DnCNN(nn.Module):
    '''
    Original DnCNN model without input conditions
    '''
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, input_x):
        out = self.dncnn(input_x)
        return out

    
class Estimation_direct(nn.Module):
    '''
    Noise estimator, with original 3 layers
    '''
    def __init__(self, input_channels = 3, output_channels = 3, num_of_layers=3):
        super(Estimation_direct, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=output_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, input):
        x = self.dncnn(input)
        return x


class DnCNN_c(nn.Module):
    def __init__(self, channels=3, num_of_layers=20, num_of_est=3):
        super(DnCNN_c, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels+ num_of_est, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x, c):
        input_x = torch.cat([x, c], dim=1)
        #input_x = x
        out = self.dncnn(input_x)
        return out  
    

class DnCNN_finetune(nn.Module):
    def __init__(self, in_ch=6, num_of_layers=17):
        super(DnCNN_finetune, self).__init__()
        self.dncnn = DnCNN2(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R')
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=3, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        input_img = torch.clamp(F.relu(self.conv1(x)), 0., 1.)
        #print(input_img.shape)
        out = self.dncnn(input_img)
        
        return out
    
    
class DecomNet(nn.Module):

    def __init__(self, layer_num=5, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.layer_num = layer_num
        self.conv0 = nn.Conv2d(4, channel, kernel_size*3, padding=4)
        feature_conv = []
        for idx in range(layer_num):
            feature_conv.append(nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size, padding=1),
                nn.ReLU()
            ))
        self.conv = nn.ModuleList(feature_conv)
        self.conv1 = nn.Conv2d(channel, 4, kernel_size, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x_max = torch.max(x, dim=1, keepdim=True)
        x = torch.cat((x, x_max[0]), dim=1)
        #x = x.permute(0, 3, 1, 2)

        out = self.conv0(x)
        for idx in range(self.layer_num):
            out = self.conv[idx](out)
        out = self.conv1(out)
        out = self.sig(out)

        r_part = out[:, 0:3, :, :]
        l_part = out[:, 3:4, :, :]

        return out, r_part, l_part
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 