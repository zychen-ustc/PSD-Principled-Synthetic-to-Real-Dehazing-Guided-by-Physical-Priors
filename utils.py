import torch
import torch.nn as nn
import torch.nn.functional as F

import os 
import cv2
import numpy as np

import random



seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
    

def pixelshuffle(image, scale):
    '''
    Discription: Given an image, return a reversible sub-sampling
    [Input]: Image ndarray float
    [Return]: A mosic image of shuffled pixels
    '''
    if scale == 1:
        return image
    w, h ,c = image.shape
    mosaic = np.array([])
    for ws in range(scale):
        band = np.array([])
        for hs in range(scale):
            temp = image[ws::scale, hs::scale, :]  #get the sub-sampled image
            band = np.concatenate((band, temp), axis = 1) if band.size else temp
        mosaic = np.concatenate((mosaic, band), axis = 0) if mosaic.size else band
    return mosaic

def reverse_pixelshuffle(image, scale, fill=0, fill_image=0, ind=[0,0]):
    '''
    Discription: Given a mosaic image of subsampling, recombine it to a full image
    [Input]: Image
    [Return]: Recombine it using different portions of pixels
    '''
    w, h, c = image.shape
    real = np.zeros((w, h, c))  #real image
    wf = 0
    hf = 0
    for ws in range(scale):
        hf = 0
        for hs in range(scale):
            temp = real[ws::scale, hs::scale, :]
            wc, hc, cc = temp.shape  #get the shpae of the current images
            if fill==1 and ws==ind[0] and hs==ind[1]:
                real[ws::scale, hs::scale, :] = fill_image[wf:wf+wc, hf:hf+hc, :]
            else:
                real[ws::scale, hs::scale, :] = image[wf:wf+wc, hf:hf+hc, :]
            hf = hf + hc
        wf = wf + wc
    return real 
