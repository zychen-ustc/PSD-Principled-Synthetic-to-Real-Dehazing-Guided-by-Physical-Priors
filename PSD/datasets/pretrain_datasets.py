import torch.utils.data as data
import os
from PIL import Image
from random import randrange
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
import torch
from utils import edge_compute

class TrainData(data.Dataset):

    def __init__(self, crop_size, train_data_dir):

        super().__init__()


        self.haze_dir = train_data_dir + 'hazy/'
        self.gt_dir = train_data_dir + 'clear/'

        self.haze_names = list(os.walk(self.haze_dir))[0][2]
        
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir


    def __getitem__(self, index):

        crop_width, crop_height =  self.crop_size
        
        haze_name = self.haze_names[index]
        gt_name = haze_name.split('_')[0] + '.jpg'
        haze_img = Image.open(self.haze_dir + haze_name).convert('RGB')
        gt_img = Image.open(self.gt_dir + gt_name).convert('RGB')

        width, height = haze_img.size

        if width < crop_width or height < crop_height:
            if width < height:
                haze_img = haze_img.resize((260, (int)(height * 260/ width)), Image.ANTIALIAS)
                gt_img = gt_img.resize((260, (int)(height * 260/ width)), Image.ANTIALIAS)
            elif width >= height:
                haze_img = haze_img.resize(((int)(width * 260/ height), 260), Image.ANTIALIAS)
                gt_img = gt_img.resize(((int)(width * 260 / height), 260), Image.ANTIALIAS)
            
            width, height = haze_img.size
        # --- random crop --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        
        transform_haze = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_gt = Compose([
            ToTensor()
        ])

        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)
        
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return haze, gt


    def __len__(self):
        return len(self.haze_names)
    
    
class TrainData_DAD(data.Dataset):

    def __init__(self, crop_size, train_data_dir):

        super().__init__()


        self.haze_dir = train_data_dir

        self.haze_names = list(os.walk(self.haze_dir))[0][2]
        #self.gt_names = np.array(list(os.walk(self.gt_dir))[0][2])
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir


    def __getitem__(self, index):

        crop_width, crop_height =  self.crop_size
        haze_name = self.haze_names[index]
        haze_gt_img = Image.open(self.haze_dir + haze_name).convert('RGB')
        
        haze_img = haze_gt_img.crop((0, 0, 400, 400))
        gt_img = haze_gt_img.crop((400, 0, 800, 400))

        width, height = haze_img.size

        if width < crop_width or height < crop_height:
            if width < height:
                haze_img = haze_img.resize((260, (int)(height * 260/ width)), Image.ANTIALIAS)
                gt_img = gt_img.resize((260, (int)(height * 260/ width)), Image.ANTIALIAS)
            elif width >= height:
                haze_img = haze_img.resize(((int)(width * 260/ height), 260), Image.ANTIALIAS)
                gt_img = gt_img.resize(((int)(width * 260 / height), 260), Image.ANTIALIAS)
            
        # --- random crop --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        
        transform_haze = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        transform_gt = Compose([
            ToTensor()
        ])

        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)
        
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return haze, gt

    def __len__(self):
        return len(self.haze_names)

    
class TrainData_GCA(data.Dataset):

    def __init__(self, crop_size, train_data_dir):

        super().__init__()


        self.haze_dir = train_data_dir + 'hazy/'
        self.gt_dir = train_data_dir + 'clear/'

        self.haze_names = list(os.walk(self.haze_dir))[0][2]
        #self.gt_names = np.array(list(os.walk(self.gt_dir))[0][2])
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir


    def __getitem__(self, index):

        crop_width, crop_height =  self.crop_size
        haze_name = self.haze_names[index]
        gt_name = haze_name.split('_')[0] + '.jpg'

        haze_img = Image.open(self.haze_dir + haze_name).convert('RGB')
        gt_img = Image.open(self.gt_dir + gt_name).convert('RGB')

        width, height = haze_img.size

        if width < crop_width or height < crop_height:
            if width < height:
                haze_img = haze_img.resize((260, (int)(height * 260/ width)), Image.ANTIALIAS)
                gt_img = gt_img.resize((260, (int)(height * 260/ width)), Image.ANTIALIAS)
            elif width >= height:
                haze_img = haze_img.resize(((int)(width * 260/ height), 260), Image.ANTIALIAS)
                gt_img = gt_img.resize(((int)(width * 260 / height), 260), Image.ANTIALIAS)
            
        # --- random crop --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        
        transform_haze = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_gt = Compose([
            ToTensor()
        ])

        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)
        haze_edge = edge_compute(haze)
        haze = torch.cat((haze, haze_edge), 0)
        
        return haze, gt


    def __len__(self):
        return len(self.haze_names)
    

class TrainData_GCAITS(data.Dataset):

    def __init__(self, crop_size, train_data_dir):

        super().__init__()


        self.haze_dir = train_data_dir + 'hazy/'
        self.gt_dir = train_data_dir + 'clear/'

        self.haze_names = list(os.walk(self.haze_dir))[0][2]
        #self.gt_names = np.array(list(os.walk(self.gt_dir))[0][2])
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir


    def __getitem__(self, index):

        crop_width, crop_height =  self.crop_size
        haze_name = self.haze_names[index]
        gt_name = haze_name.split('_')[0] + '.png'

        haze_img = Image.open(self.haze_dir + haze_name).convert('RGB')
        gt_img = Image.open(self.gt_dir + gt_name).convert('RGB')

        width, height = haze_img.size

        if width < crop_width or height < crop_height:
            if width < height:
                haze_img = haze_img.resize((260, (int)(height * 260/ width)), Image.ANTIALIAS)
                gt_img = gt_img.resize((260, (int)(height * 260/ width)), Image.ANTIALIAS)
            elif width >= height:
                haze_img = haze_img.resize(((int)(width * 260/ height), 260), Image.ANTIALIAS)
                gt_img = gt_img.resize(((int)(width * 260 / height), 260), Image.ANTIALIAS)
            
        # --- random crop --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        
        transform_haze = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_gt = Compose([
            ToTensor()
        ])

        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)
        haze_edge = edge_compute(haze)
        haze = torch.cat((haze, haze_edge), 0)

        return haze, gt


    def __len__(self):
        return len(self.haze_names)
    
    
    
class TrainData_DADGCA(data.Dataset):

    def __init__(self, crop_size, train_data_dir):

        super().__init__()


        self.haze_dir = train_data_dir

        self.haze_names = list(os.walk(self.haze_dir))[0][2]
        #self.gt_names = np.array(list(os.walk(self.gt_dir))[0][2])
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir


    def __getitem__(self, index):

        crop_width, crop_height =  self.crop_size
        haze_name = self.haze_names[index]
        haze_gt_img = Image.open(self.haze_dir + haze_name).convert('RGB')
        
        haze_img = haze_gt_img.crop((0, 0, 400, 400))
        gt_img = haze_gt_img.crop((400, 0, 800, 400))

        width, height = haze_img.size

        if width < crop_width or height < crop_height:
            if width < height:
                haze_img = haze_img.resize((260, (int)(height * 260/ width)), Image.ANTIALIAS)
                gt_img = gt_img.resize((260, (int)(height * 260/ width)), Image.ANTIALIAS)
            elif width >= height:
                haze_img = haze_img.resize(((int)(width * 260/ height), 260), Image.ANTIALIAS)
                gt_img = gt_img.resize(((int)(width * 260 / height), 260), Image.ANTIALIAS)
            
        # --- random crop --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        
        transform_haze = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_gt = Compose([
            ToTensor()
        ])

        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)
        edge = edge_compute(haze)
        haze = torch.cat((haze, edge), 0)

        return haze, gt


    def __len__(self):
        return len(self.haze_names)

    
class ValData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()

        self.haze_dir = val_data_dir + 'haze/'
        self.gt_dir = val_data_dir + 'clear/'
        self.haze_names = np.array(list(os.walk(self.haze_dir))[0][2])


    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = haze_name.split('_')[0] + '.png'
        haze_img = Image.open(self.haze_dir + haze_name)
        gt_img = Image.open(self.gt_dir + gt_name)

        haze_reshaped = haze_img
        haze_reshaped = haze_reshaped.resize((256, 256), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        haze_reshaped = transform_haze(haze_reshaped)
        gt = transform_gt(gt_img)
        
        return haze, haze_reshaped, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
    
    
class ValData_GCA(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()

        self.haze_dir = val_data_dir + 'haze/'
        self.gt_dir = val_data_dir + 'clear/'
        self.haze_names = np.array(list(os.walk(self.haze_dir))[0][2])


    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = haze_name.split('_')[0] + '.png'
        haze_img = Image.open(self.haze_dir + haze_name)
        gt_img = Image.open(self.gt_dir + gt_name)

        haze_reshaped = haze_img
        haze_reshaped = haze_reshaped.resize((256, 256), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        haze_reshaped = transform_haze(haze_reshaped)
        
        gt = transform_gt(gt_img)
        haze_edge = edge_compute(haze)
        haze = torch.cat((haze, haze_edge), 0)
        return haze, haze_reshaped, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
    

class TestData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()

        self.haze_dir = val_data_dir
        #self.gt_dir = val_data_dir + 'clear/'
        with open('/code/PSD/configs/record.txt', "r") as file:
            self.haze_names = file.readlines()


    def get_images(self, index):
        haze_name = self.haze_names[index][:-1]
        haze_img = Image.open(self.haze_dir + haze_name).convert('RGB')
        
        haze_reshaped = haze_img
        haze_reshaped = haze_reshaped.resize((256, 256), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        haze_reshaped = transform_haze(haze_reshaped)

        return haze, haze_reshaped, haze_name

    
    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
    
    
class TestData2(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()

        self.haze_dir = val_data_dir
        self.haze_names = list(os.walk(self.haze_dir))[0][2]


    def get_images(self, index):
        haze_name = self.haze_names[index]
        haze_img = Image.open(self.haze_dir + haze_name).convert('RGB')
        
        haze_reshaped = haze_img
        haze_reshaped = haze_reshaped.resize((256, 256), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        haze_reshaped = transform_haze(haze_reshaped)

        return haze, haze_reshaped, haze_name
    

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
    
    
    
class TestData_GCA(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()

        self.haze_dir = val_data_dir
        #self.gt_dir = val_data_dir + 'clear/'
        with open('/code/PSD/configs/record.txt', "r") as file:
            self.haze_names = file.readlines()


    def get_images(self, index):
        haze_name = self.haze_names[index][:-1]
        #gt_name = haze_name.split('_')[0] + '.png'
        haze_img = Image.open(self.haze_dir + haze_name).convert('RGB')
        
        width, height = haze_img.size
        im_w, im_h = haze_img.size
        if im_w % 4 != 0 or im_h % 4 != 0:
            haze_img = haze_img.resize((int(im_w // 4 * 4), int(im_h // 4 * 4))) 
        img = np.array(haze_img).astype('float') 
        img_data = torch.from_numpy(img.transpose((2, 0, 1))).float()
        edge_data = edge_compute(img_data)
        in_data = torch.cat((img_data, edge_data), dim=0) - 128 
        return in_data, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
    

class TestData_FFA(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()

        self.haze_dir = val_data_dir
        self.haze_names = list(os.walk(self.haze_dir))[0][2]
        #self.gt_dir = val_data_dir + 'clear/'
        with open('/code/PSD/configs/record.txt', "r") as file:
            self.haze_names = file.readlines()


    def get_images(self, index):
        haze_name = self.haze_names[index][:-1]
        #gt_name = haze_name.split('_')[0] + '.png'
        haze_img = Image.open(self.haze_dir + haze_name).convert('RGB')
        haze_reshaped = haze_img
        haze_reshaped = haze_reshaped.resize((256, 256), Image.ANTIALIAS)
        #gt_img = Image.open(self.gt_dir + gt_name)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        haze_reshaped = transform_haze(haze_reshaped)
        #haze_edge_data = edge_compute(haze)
        #haze = torch.cat((haze, haze_edge_data), 0)
        #gt = transform_gt(gt_img)

        return haze, haze_reshaped, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
    
    
    
def edge_compute(x):
    
    x_diffx = torch.abs(x[:,:,1:] - x[:,:,:-1])
    x_diffy = torch.abs(x[:,1:,:] - x[:,:-1,:])

    y = x.new(x.size())
    y.fill_(0)
    y[:,:,1:] += x_diffx
    y[:,:,:-1] += x_diffx
    y[:,1:,:] += x_diffy
    y[:,:-1,:] += x_diffy
    y = torch.sum(y,0,keepdim=True)/3
    y /= 4
    
    return y
