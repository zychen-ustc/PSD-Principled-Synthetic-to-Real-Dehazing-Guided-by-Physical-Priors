import torch
import torch.nn as nn
import torch.nn.functional as F


class tv_loss_f(nn.Module):

    def __init__(self):

        super(TVLoss, self).__init__()

        self.e = 0.000001 ** 2


    def forward(self, x):

        batch_size = x.size()[0]

        h_tv = torch.abs((x[:, :, 1:, :]-x[:, :, :-1, :]))

        h_tv = torch.mean(torch.sqrt(h_tv ** 2 + self.e))

        w_tv = torch.abs((x[:, :, :, 1:]-x[:, :, :, :-1]))

        w_tv = torch.mean(torch.sqrt(w_tv ** 2 + self.e))

        return h_tv + w_tv
    

def get_SV_from_HSV(img):
    
    if (img.max(1)[0] == 0).all():
        return 0, 0
    
    else:
        S = (img.max(1)[0] - img.min(1)[0]) / img.max(1)[0]
        V = img.max(1)[0]

        S[img.max(1)[0] == 0] = 0

        return S, V


def get_HSV(img):
    
    hue = torch.zeros((img.shape[0], img.shape[2], img.shape[3])).to(img.device)

    hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0]))[
        img[:, 2] == img.max(1)[0]]
    hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0]))[
        img[:, 1] == img.max(1)[0]]
    hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0]))[
        img[:, 0] == img.max(1)[0]]) % 6

    hue[img.min(1)[0] == img.max(1)[0]] = 0.0
    hue[hue < 0] += 6

    hue = hue / 6
    '''    
    if (img.max(1)[0] == 0).all():
        return 0, 0
    else:
        S = (img.max(1)[0] - img.min(1)[0]) / img.max(1)[0]
        V = img.max(1)[0]

        S[img.max(1)[0] == 0] = 0
    '''
    return hue


def get_dark_channel(I, w):
    
    _, _, H, W = I.shape
    maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
    dc = maxpool(0 - I[:, :, :, :])

    return -dc


def get_bright_channel(I, w):
    
    _, _, H, W = I.shape
    maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
    bc = maxpool(I[:, :, :, :])

    return bc


def get_atmosphere(I, dark_ch, p):
    
    B, _, H, W = dark_ch.shape
    num_pixel = int(p * H * W)
    flat_dc = dark_ch.resize(B, H * W)
    flat_I = I.resize(B, 3, H * W)
    index = torch.argsort(flat_dc, descending=True)[:, :num_pixel]
    A = torch.zeros((B, 3)).to('cuda')
    
    for i in range(B):
        A[i] = flat_I[i, :, index].mean((1, 2))

    return A


def get_atmosphere2(I, bright_ch, p):
    
    B, _, H, W = bright_ch.shape
    num_pixel = int(p * H * W)
    flat_bc = bright_ch.resize(B, H * W)
    flat_I = I.resize(B, 3, H * W)
    index = torch.argsort(flat_bc, descending=False)[:, :num_pixel]
    A = torch.zeros((B, 3)).to('cuda')
    
    for i in range(B):
        A[i] = flat_I[i, :, index].mean((1, 2))

    return A


def regular_loss(J):
    
    if len(J[J < 0]) == 0:
        loss2 = 0
        
    else:
        loss2 = abs(J[J < 0].sum() / len(J[J < 0]))
        
    if len(J[J > 1]) == 0:
        loss3 = 0
        
    else:
        loss3 = abs(J[J > 1].sum() / len(J[J > 1]))

    loss = loss2 + loss3
    
    return loss


def get_cap_loss(img, T):
    
    maxpool = nn.MaxPool2d(15, stride=1, padding=15 // 2)

    s, v = get_SV_from_HSV(img)
    sigma = 0.041337
    sigmaMat = torch.normal(0, sigma, size=(B, 1, H, W)).to('cuda')

    depth = 0.121779 + 0.959710 * v - 0.780245 * s + sigmaMat
    depth_R = -maxpool(-depth)

    img_gray = (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2])[:, None]
    refinedDR = GuidedFilter(60, 0.001)(img_gray, depth_R)
    t_slide = torch.exp(-refinedDR).clamp(0.05, 1)

    loss = F.smooth_l1_loss(T, t_slide)
    
    return loss


def max_contrast(J, I):
    
    dc = get_dark_channel(I, 25)
    A = get_atmosphere(I, dc, 0.0001)
    gamma = (3 * A) / A.sum()
    J_norm = J / A[:, :, None, None]
    loss = tv_loss_f()(J)

    return loss


def bright_channel(img, T, w=35, p=0.0001):
    
    bright_channel = get_bright_channel(img, w)
    # dc = get_dark_channel(img, 15)

    A = get_atmosphere2(img, bright_channel, p)

    norm_I = (1 - img) / (1 - A[:, :, None, None] + 1e-6)
    bright_channel = get_bright_channel(norm_I, w)
    t_slide = (1 - 0.95 * bright_channel)

    # t1 = t_slide[dc < 0.6]
    # t2 = T[dc < 0.6]
    # if len(t1) == 0:
    #    return 0
    # print(T.shape, t1.shape)
    loss = F.smooth_l1_loss(T, t_slide)
    
    return loss


def dark_channel(img, T, w=35, p=0.0001):
    
    dark_channel = get_dark_channel(img, w)
    A = get_atmosphere(img, dark_channel, p)

    norm_I = (1 - img) / (1 - A[:, :, None, None] + 1e-6)
    dark_channel = get_dark_channel(norm_I, w)
    t_slide = (1 - 0.95 * dark_channel)

    loss = F.smooth_l1_loss(T, t_slide)
    
    return loss


def attention_bc_loss(J, img):
    
    maxpool = nn.MaxPool2d(15, stride=1, padding=15 // 2)

    s, v = get_SV_from_HSV(img)
    sigma = 0.041337
    sigmaMat = torch.normal(0, sigma, size=(B, 1, H, W)).to('cuda')

    depth = 0.121779 + 0.959710 * v - 0.780245 * s + sigmaMat
    depth_R = -maxpool(-depth)

    img_gray = (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2])[:, None]
    refinedDR = GuidedFilter(60, 0.001)(img_gray, depth_R)

    print(refinedDR.max(), refinedDR.shape)

    bc = get_bright_channel(J)
    norm_bc = bc * refinedDR

    target = torch.ones(norm_bc.shape)

    loss = F.smooth_l1_loss(norm_bc, target)
    
    return loss


def saturation_loss(img, T, w=15, p=0.0001):
    
    dark_channel = get_dark_channel(img, w)
    A = get_atmosphere(img, dark_channel, p)
    A = A[:, :, None, None]

    t_min = ((img - A) / (1 - A)).max(1)[0][:, None]
    t1 = T[T < t_min]
    
    if len(t1) == 0:
        return 0
    
    t2 = t_min[T < t_min]
    loss = F.mse_loss(t1, t2)
    
    return loss


def lwf_sky(img, J, J_o, w=15):
    
    dc = get_dark_channel(img, w)
    dc_shaped = dc.repeat(1, 3, 1, 1)
    J_1 = J[dc_shaped > 0.6]
    J_2 = J_o[dc_shaped > 0.6]
    
    if len(J_1) == 0:
        return 0

    loss = F.smooth_l1_loss(J_1, J_2)
    
    return loss


def retinex_loss(J, img1, img2):
    
    s1, _ = get_SV_from_HSV(img1)
    s2, _ = get_SV_from_HSV(img2)
    J1 = J[(s1.mean((1, 2)) < s2.mean((1, 2)))]
    J2 = img2[(s1.mean((1, 2)) < s2.mean((1, 2)))]
    
    if len(J1) == 0:
        return 0
    
    loss = F.smooth_l1_loss(J1, J2)

    return loss


def get_luminance(img):
    
    Y = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    maxpool = nn.MaxPool2d(15, stride=1, padding=15 // 2)
    dc = -maxpool(-Y[:, None])
    
    target = torch.zeros(dc.shape).to('cuda')
    loss = F.smooth_l1_loss(dc, target)
    
    return loss

'''
def GB_loss(img, T):
    R = (img[:, 0])[:, None]
    GB = img[:, 1:].max(1)[0][:, None]
    maxpool = nn.MaxPool2d(15, stride=1, padding=15//2)
    max_R = maxpool(R)
    max_GB = maxpool(GB)
    D = max_R - max_GB
    B = (img[:, :, D.argmin() // img.shape])
'''

def DCLoss(img, patch_size):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size//2, patch_size//2))
    dc = maxpool(0-img[:, None, :, :, :])
    
    target = Variable(torch.FloatTensor(dc.shape).zero_().cuda()) 
     
    loss = L1Loss(size_average=True)(-dc, target)
    
    return loss

def BCLoss(img, patch_size):
    """
    calculating bright channel of image, the image shape is of N*C*W*H
    """
    patch_size = 35
    dc = maxpool(img[:, None, :, :, :])
    
    target = Variable(torch.FloatTensor(dc.shape).zero_().cuda()+1) 
    loss = L1Loss(size_average=False)(dc, target)
    
    return loss
    