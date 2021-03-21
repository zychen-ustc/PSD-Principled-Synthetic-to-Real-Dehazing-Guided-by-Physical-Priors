import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tfs
from torch.autograd import Variable


class energy_dc_loss(nn.Module):
    
    def __init__(self, w=15, p=0.001, omega=0.95, eps=1e-5, lambda1=2, lambda2=1e-2):
        super(energy_dc_loss, self).__init__()

        self.w = w
        self.p = p
        self.omega = omega
        self.eps = eps
        self.param = [lambda1, lambda2]

    def forward(self, img, y_pred):

        dark_channel = self.get_dark_channel(img, self.w)
        A = self.get_atmosphere(img, dark_channel, self.p)

        normI = img.permute(2, 3, 0, 1)
        normI = (normI / A).permute(2, 3, 0, 1)
        norm_dc = self.get_dark_channel(normI, self.w)

        t_slide = (1 - self.omega * norm_dc) # transmission map predicted by dark channel prior

        patches_I = F.unfold(y_pred, (3, 3))
        Y_I = patches_I.repeat([1, 9, 1])
        Y_J = patches_I.repeat_interleave(9, 1)

        temp = F.unfold(img, (3, 3))
        B, N = temp.shape[0], temp.shape[2]
        
        img_patches = temp.view(B, 3, 9, N).permute(0, 3, 2, 1)
        mean_patches = torch.mean(img_patches, 2, True)
        
        XX_T = (torch.matmul(img_patches.permute(0, 1, 3, 2), img_patches) / 9)
        UU_T = torch.matmul(mean_patches.permute(0, 1, 3, 2), mean_patches)
        var_patches = XX_T - UU_T
        
        matrix_to_invert = (self.eps / 9) * torch.eye(3).to('cuda') + var_patches
        var_fac = torch.inverse(matrix_to_invert)
        
        weights = torch.matmul(img_patches - mean_patches, var_fac)
        weights = torch.matmul(weights, (img_patches - mean_patches).permute(0, 1, 3, 2)) + 1
        weights = weights / 9
        weights = weights.view(-1, N, 81)

        neighbour_difference = (Y_I - Y_J) ** 2
        fidelity_term = torch.matmul(neighbour_difference, weights).sum() # data fidelity
        prior_term = ((y_pred - t_slide) ** 2).sum() # regularization

        loss = (self.param[0] * (1/2) * fidelity_term + self.param[1] * prior_term) / N

        return loss
    
    
    def get_dark_channel(self, I, w):
        
        _, _, H, W = I.shape
        maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
        dc = maxpool(0 - I[:, :, :, :])

        return -dc

    
    def get_atmosphere(self, I, dark_ch, p):
        
        B, _, H, W = dark_ch.shape
        num_pixel = int(p * H * W)
        flat_dc = dark_ch.resize(B, H * W)
        flat_I = I.resize(B, 3, H * W)
        index = torch.argsort(flat_dc, descending=True)[:, :num_pixel]
        A = torch.zeros((B, 3)).to('cuda')
        
        for i in range(B):
            A[i] = flat_I[i, :, index[i][torch.argsort(torch.max(flat_I[i][:, index[i]], 0)[0], descending=True)[0]]]

        return A


class energy_bc_loss(nn.Module):
    
    def __init__(self, w=15, p=0.001, omega=0.95, eps=1e-5, lambda1=2, lambda2=1):
        super(energy_bc_loss, self).__init__()

        self.w = w
        self.p = p
        self.omega = omega
        self.eps = eps
        self.param = [lambda1, lambda2]

    def forward(self, img, y_pred):

        bright_channel = self.get_bright_channel(img, self.w)
        A = self.get_atmosphere(img, bright_channel, self.p)
        
        #bc = maxpool((1 - I[:, :, :, :]) / (1 - A[:, :, None, None] + self.eps))
        norm_I = (1 - img) / (1 - A[:, :, None, None] + self.eps)
        bright_channel = self.get_bright_channel(norm_I, self.w)

        t_slide = (1 - bright_channel)

        patches_I = F.unfold(y_pred, (3, 3))
        Y_I = patches_I.repeat([1, 9, 1])
        Y_J = patches_I.repeat_interleave(9, 1)

        temp = F.unfold(img, (3, 3))
        B, N = temp.shape[0], temp.shape[2]
        
        img_patches = temp.view(B, 3, 9, N).permute(0, 3, 2, 1)
        mean_patches = torch.mean(img_patches, 2, True)
        
        XX_T = (torch.matmul(img_patches.permute(0, 1, 3, 2), img_patches) / 9)
        UU_T = torch.matmul(mean_patches.permute(0, 1, 3, 2), mean_patches)
        var_patches = XX_T - UU_T
        
        matrix_to_invert = (self.eps / 9) * torch.eye(3).to('cuda') + var_patches
        var_fac = torch.inverse(matrix_to_invert)
        
        weights = torch.matmul(img_patches - mean_patches, var_fac)
        weights = torch.matmul(weights, (img_patches - mean_patches).permute(0, 1, 3, 2)) + 1
        weights = weights / 9
        weights = weights.view(-1, N, 81)

        neighbour_difference = (Y_I - Y_J) ** 2
        smoothness_term = torch.matmul(neighbour_difference, weights).sum()
        data_term = ((y_pred - t_slide) ** 2).sum()

        loss = (smoothness_term + 0.001 * data_term) / N

        return loss

    
    def get_bright_channel(self, I, w):
        
        _, _, H, W = I.shape
        maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
        bc = maxpool(I[:, :, :, :])
        
        return bc

    
    def get_dark_channel(self, I, w):
        
        _, _, H, W = I.shape
        maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
        dc = maxpool(0 - I[:, :, :, :])

        return -dc

    
    def get_atmosphere(self, I, bright_ch, p):
        
        B, _, H, W = bright_ch.shape
        num_pixel = int(p * H * W)
        flat_bc = bright_ch.resize(B, H * W)
        flat_I = I.resize(B, 3, H * W)
        index = torch.argsort(flat_bc, descending=False)[:, :num_pixel]
        A = torch.zeros((B, 3)).to('cuda')
        
        for i in range(B):
            A[i] = flat_I[i, :, index].mean((1, 2))

        return A
    
class energy_dc_bc_loss(nn.Module):
    
    def __init__(self, k, w=15, p=0.001, omega=0.95, eps=1e-5, lambda1=2, lambda2=1e-4):
        super(double, self).__init__()

        self.w = w
        self.p = p
        self.k = k
        self.omega = omega
        self.eps = eps
        self.param = [lambda1, lambda2]

        
    def forward(self, img, y_pred):

        dark_channel = self.get_dark_channel(img, self.w)
        A_1 = self.get_atmosphere1(img, dark_channel, self.p)
        normI = img.permute(2, 3, 0, 1)
        normI = (normI / A_1).permute(2, 3, 0, 1)
        norm_dc = self.get_dark_channel(normI, self.w)
        t_slide1 = (1 - self.omega * norm_dc)
        
        
        bright_channel = self.get_bright_channel(img, self.w)
        A_2 = self.get_atmosphere2(img, bright_channel, self.p)
        norm_I = (1 - img) / (1 - A_2[:, :, None, None])
        bright_channel = self.get_bright_channel(norm_I, self.w)
        t_slide2 = (1 - bright_channel)
        
        
        patches_I = F.unfold(y_pred, (3, 3))
        Y_I = patches_I.repeat([1, 9, 1])
        Y_J = patches_I.repeat_interleave(9, 1)

        temp = F.unfold(img, (3, 3))
        B, N = temp.shape[0], temp.shape[2]
        
        img_patches = temp.view(B, 3, 9, N).permute(0, 3, 2, 1)
        mean_patches = torch.mean(img_patches, 2, True)
        
        XX_T = (torch.matmul(img_patches.permute(0, 1, 3, 2), img_patches) / 9)
        UU_T = torch.matmul(mean_patches.permute(0, 1, 3, 2), mean_patches)
        var_patches = XX_T - UU_T
        
        matrix_to_invert = (self.eps / 9) * torch.eye(3).to('cuda') + var_patches
        var_fac = torch.inverse(matrix_to_invert)
        
        weights = torch.matmul(img_patches - mean_patches, var_fac)
        weights = torch.matmul(weights, (img_patches - mean_patches).permute(0, 1, 3, 2)) + 1
        weights = weights / 9
        weights = weights.view(-1, N, 81)

        neighbour_difference = (Y_I - Y_J) ** 2
        smoothness_term = torch.matmul(neighbour_difference, weights).sum()
        data_term1 = ((y_pred - t_slide1) ** 2).sum()
        data_term2 = ((y_pred - t_slide2) ** 2).sum()

        loss1 = (self.param[0] * (1/2) * smoothness_term + self.param[1] * data_term1) / N
        loss2 = (0.01 * smoothness_term + data_term2) / N
        loss = loss1 + self.k * loss2

        return loss

    
    def get_dark_channel(self, I, w):
        
        _, _, H, W = I.shape
        maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
        dc = maxpool(0 - I[:, :, :, :])

        return -dc
    
    
    def get_bright_channel(self, I, w):
        
        _, _, H, W = I.shape
        maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
        bc = maxpool(I[:, :, :, :])
        
        return bc
    

    def get_atmosphere1(self, I, dark_ch, p):
        
        B, _, H, W = dark_ch.shape
        num_pixel = int(p * H * W)
        flat_dc = dark_ch.resize(B, H * W)
        flat_I = I.resize(B, 3, H * W)
        index = torch.argsort(flat_dc, descending=True)[:, :num_pixel]
        A = torch.zeros((B, 3)).to('cuda')
        for i in range(B):
            A[i] = flat_I[i, :, index[i][torch.argsort(torch.max(flat_I[i][:, index[i]], 0)[0], descending=True)[0]]]

        return A
    
    
    def get_atmosphere2(self, I, bright_ch, p):
        
        B, _, H, W = bright_ch.shape
        num_pixel = int(p * H * W)
        flat_bc = bright_ch.resize(B, H * W)
        flat_I = I.resize(B, 3, H * W)
        index = torch.argsort(flat_bc, descending=False)[:, :num_pixel]
        A = torch.zeros((B, 3)).to('cuda')
        
        for i in range(B):
            A[i] = flat_I[i, :, index].mean((1, 2))

        return A
    
    
class energy_cap_loss(nn.Module):
    
    def __init__(self, w=15, p=0.001, omega=0.95, eps=1e-5, lambda1=2, lambda2=1e-4):
        super(energy_cap_loss, self).__init__()

        self.w = w
        self.p = p
        self.omega = omega
        self.eps = eps
        self.param = [lambda1, lambda2]

    def forward(self, img, y_pred):
        
        B, C, H, W = img.shape
        s, v = self.get_SV_from_HSV(img)
        s = s[:, None]
        v = v[:, None]
        sigma = 0.041337
        sigmaMat = torch.normal(0, sigma, size=(B, 1, H, W)).to('cuda')
        
        depth = 0.121779 + 0.959710 * v - 0.780245 * s + sigmaMat

        maxpool = nn.MaxPool2d(self.w, stride=1, padding=self.w // 2)
        depth_R = -maxpool(-depth)
        
        img_gray = (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2])[:, None]
        refinedDR = GuidedFilter(60, 0.001)(img_gray, depth_R)
        t_slide = torch.exp(-refinedDR).clamp(0.05, 1)
        patches_I = F.unfold(y_pred, (3, 3))
        Y_I = patches_I.repeat([1, 9, 1])
        Y_J = patches_I.repeat_interleave(9, 1)

        temp = F.unfold(img, (3, 3))
        B, N = temp.shape[0], temp.shape[2]
        
        img_patches = temp.view(B, 3, 9, N).permute(0, 3, 2, 1)
        mean_patches = torch.mean(img_patches, 2, True)
        
        XX_T = (torch.matmul(img_patches.permute(0, 1, 3, 2), img_patches) / 9)
        UU_T = torch.matmul(mean_patches.permute(0, 1, 3, 2), mean_patches)
        var_patches = XX_T - UU_T
        
        matrix_to_invert = (self.eps / 9) * torch.eye(3).to('cuda') + var_patches
        var_fac = torch.inverse(matrix_to_invert)
        
        weights = torch.matmul(img_patches - mean_patches, var_fac)
        weights = torch.matmul(weights, (img_patches - mean_patches).permute(0, 1, 3, 2)) + 1
        weights = weights / 9
        weights = weights.view(-1, N, 81)

        neighbour_difference = (Y_I - Y_J) ** 2
        smoothness_term = torch.matmul(neighbour_difference, weights).sum()
        data_term = ((y_pred - t_slide) ** 2).sum()

        loss = (smoothness_term + 0.01 * data_term) / N

        return loss
    
    
    def get_SV_from_HSV(self, img):
        
        img = img.clamp(-0.0001, 1)
        
        if (img.max(1)[0] == 0).all():
            return 0, 0
        
        else:
            S = (img.max(1)[0] - img.min(1)[0]) / img.max(1)[0]
            V = img.max(1)[0]

            S[img.max(1)[0] == 0] = 0

            return S, V
        

def diff_x(input, r):
    
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output


def diff_y(input, r):
    
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output


class BoxFilter(nn.Module):
    
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)

    
class GuidedFilter(nn.Module):
    
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b
    
    
class energy_dc_loss_edge(nn.Module):
    
    def __init__(self, w=15, p=0.001, omega=0.95, eps=1e-5, lambda1=2, lambda2=1e-2):
        super(energy_dc_loss_edge, self).__init__()

        self.w = w
        self.p = p
        self.omega = omega
        self.eps = eps
        self.param = [lambda1, lambda2]

    def forward(self, img, y_pred):

        dark_channel = self.get_dark_channel_edge(img, self.w)
        A = self.get_atmosphere(img, dark_channel, self.p)

        normI = img.permute(2, 3, 0, 1)
        normI = (normI / A).permute(2, 3, 0, 1)
        norm_dc = self.get_dark_channel_edge(normI, self.w)

        t_slide = (1 - self.omega * norm_dc)

        patches_I = F.unfold(y_pred, (3, 3))
        Y_I = patches_I.repeat([1, 9, 1])
        Y_J = patches_I.repeat_interleave(9, 1)

        temp = F.unfold(img, (3, 3))
        B, N = temp.shape[0], temp.shape[2]
        
        img_patches = temp.view(B, 3, 9, N).permute(0, 3, 2, 1)
        mean_patches = torch.mean(img_patches, 2, True)
        
        XX_T = (torch.matmul(img_patches.permute(0, 1, 3, 2), img_patches) / 9)
        UU_T = torch.matmul(mean_patches.permute(0, 1, 3, 2), mean_patches)
        var_patches = XX_T - UU_T
        
        matrix_to_invert = (self.eps / 9) * torch.eye(3).to('cuda') + var_patches
        var_fac = torch.inverse(matrix_to_invert)
        
        weights = torch.matmul(img_patches - mean_patches, var_fac)
        weights = torch.matmul(weights, (img_patches - mean_patches).permute(0, 1, 3, 2)) + 1
        weights = weights / 9
        weights = weights.view(-1, N, 81)

        neighbour_difference = (Y_I - Y_J) ** 2
        smoothness_term = torch.matmul(neighbour_difference, weights).sum()
        data_term = ((y_pred - t_slide) ** 2).sum()

        loss = (self.param[0] * (1/2) * smoothness_term + self.param[1] * data_term) / N

        return loss
    
    
    def get_dark_channel_edge(self, I, w):
        
        avr = I.mean()
        std = I.std()
        mean_map = I.mean(dim=1)[:, None]
        skewness = ((I - avr)**3) / (I.numel() * (std**3))
        idx1 = (mean_map <= avr)
        idx2 = (mean_map > avr)
        dc_temp = self.get_dark_channel(I, w)
        dc = torch.zeros(mean_map.shape)[:, None]
        
        print("DC:", dc.shape, "DC_TEMP: ", dc_temp.shape, "idx1, idx2:", idx1.shape, idx2.shape)
        
        dc[mean_map <= avr] = dc_temp ** (1 + avr)
        dc[mean_map > avr] = (1 - std) * dc_temp - skewness * std
        
        return dc
    
    def get_dark_channel(self, I, w):
        
        _, _, H, W = I.shape
        maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
        dc = maxpool(0 - I[:, :, :, :])
        
        return -dc
    
    '''
    def get_edge_dark_channel(self, I, w):
        avr = I.mean()
        epsilon = I.std()
        Rho = ((I - avr)**3).sum() / (I.numel() * (epsilon ** 3))
        dc = torch.zeros((I.shape[0], 1, I.shape[2], I.shape[3]))
        dc[]
    '''
    def get_atmosphere(self, I, dark_ch, p):
        
        B, _, H, W = dark_ch.shape
        num_pixel = int(p * H * W)
        flat_dc = dark_ch.resize(B, H * W)
        flat_I = I.resize(B, 3, H * W)
        index = torch.argsort(flat_dc, descending=True)[:, :num_pixel]
        A = torch.zeros((B, 3)).to('cuda')
        
        for i in range(B):
            A[i] = flat_I[i, :, index[i][torch.argsort(torch.max(flat_I[i][:, index[i]], 0)[0], descending=True)[0]]]

        return A