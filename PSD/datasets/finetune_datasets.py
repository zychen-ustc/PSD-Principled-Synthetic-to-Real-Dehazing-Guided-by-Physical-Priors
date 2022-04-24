import torch.utils.data as data
import os
from PIL import Image
from random import randrange
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize


class RealTrainData(data.Dataset):

    def __init__(self, crop_size, train_data_dir):

        super().__init__()

        #haze_dir1 = train_data_dir + 'unlabeled1/'
        haze_dir1 = train_data_dir
        #gt_dir1 = train_data_dir + 'clear1/'
        
        #self.haze_names = [haze_dir1 + name[i] for i in range(len(name))]
        #self.gt_names = [gt_dir1 + name[i].split('.')[0] + '_dehaze.png' for i in range(len(name))]

        self.haze_names = list(os.walk(haze_dir1))[0][2]
        #self.gt_names = np.array(list(os.walk(gt_dir1))[0][2])[index]
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir
  


    def __getitem__(self, index):
        crop_width, crop_height =  self.crop_size
        haze_name = self.train_data_dir + self.haze_names[index]
        #gt_name = self.gt_names[index]

        haze_img = Image.open(haze_name).convert('RGB')
        #gt_img = Image.open(gt_name).convert('RGB')

        width, height = haze_img.size
        
        transform_haze = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_gt = Compose([
            ToTensor()
        ])
        '''
        if width < crop_width or height < crop_height:
            print("IN")
            if width < height:
                print('TYPEI')
                new_width = 260
                new_height = int(260 * (height / width))
            elif width >= height:
                print("TYPEII")
                new_height = 260
                new_width = int(260 * (width / height))

            haze_img = haze_img.resize((new_width, new_height), Image.ANTIALIAS)
            width, height = haze_img.size
            print(width, height)
        '''
        if width < crop_width or height < crop_height:
            if width < height:
                haze_img = haze_img.resize((260, int(260 * (height / width))), Image.ANTIALIAS)
            else:
                haze_img = haze_img.resize((int(260 * (width / height)), 260), Image.ANTIALIAS)
            width, height = haze_img.size
        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        #gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        

        haze = transform_haze(haze_crop_img)
        #gt = transform_gt(gt_crop_img)

        #if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
        #    raise Exception('Bad image channel: {}'.format(gt_name))

        #return haze, gt
        return haze, haze_name


    def __len__(self):
        return len(self.haze_names)
    

class RealTrainData_pseudo_gt(data.Dataset):

    def __init__(self, crop_size, train_data_dir):

        super().__init__()
        
        self.haze_data_dir = train_data_dir + 'unlabeled/'
        self.gt_data_dir = train_data_dir + 'retinex/'
        self.haze_names = list(os.walk(self.haze_data_dir))[0][2]
        
        self.crop_size = crop_size
        

    def __getitem__(self, index):
        crop_width, crop_height =  self.crop_size
        haze_name = self.haze_data_dir + self.haze_names[index]
        gt_name = self.gt_data_dir + self.haze_names[index].split('.')[0] + '.png'

        haze_img = Image.open(haze_name).convert('RGB')
        gt_img = Image.open(gt_name).convert('RGB')

        width, height = haze_img.size
        
        transform_haze = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_gt = Compose([
            ToTensor()
        ])
        '''
        if width < crop_width or height < crop_height:
            print("IN")
            if width < height:
                print('TYPEI')
                new_width = 260
                new_height = int(260 * (height / width))
            elif width >= height:
                print("TYPEII")
                new_height = 260
                new_width = int(260 * (width / height))

            haze_img = haze_img.resize((new_width, new_height), Image.ANTIALIAS)
            width, height = haze_img.size
            print(width, height)
        '''
        if width < crop_width or height < crop_height:
            if width < height:
                haze_img = haze_img.resize((260, int(260 * (height / width))), Image.ANTIALIAS)
                gt_img = gt_img.resize((260, int(260 * (height / width))), Image.ANTIALIAS)
            else:
                haze_img = haze_img.resize((int(260 * (width / height)), 260), Image.ANTIALIAS)
                gt_img = gt_img.resize((int(260 * (width / height)), 260), Image.ANTIALIAS)
            width, height = haze_img.size
        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        

        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)

        #if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
        #    raise Exception('Bad image channel: {}'.format(gt_name))

        #return haze, gt
        return haze, gt, haze_name


    def __len__(self):
        return len(self.haze_names)
    
    
class RealTrainData_CLAHE(data.Dataset):

    def __init__(self, crop_size, train_data_dir):

        super().__init__()
        
        self.haze_data_dir = train_data_dir + 'unlabeled/'
        self.gt_data_dir = train_data_dir + 'CLAHE_3/'
        with open('/code/dehazeproject/configs/Correct_train_record.txt', "r") as file:
            self.haze_names = file.readlines()
        
        self.crop_size = crop_size
        

    def __getitem__(self, index):
        crop_width, crop_height =  self.crop_size
        haze_name = self.haze_data_dir + self.haze_names[index][:-1]
        gt_name = self.gt_data_dir + self.haze_names[index][:-1].split('.')[0] + '.png'

        haze_img = Image.open(haze_name).convert('RGB')
        gt_img = Image.open(gt_name).convert('RGB')

        width, height = haze_img.size
        
        transform_haze = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_gt = Compose([
            ToTensor()
        ])
        '''
        if width < crop_width or height < crop_height:
            print("IN")
            if width < height:
                print('TYPEI')
                new_width = 260
                new_height = int(260 * (height / width))
            elif width >= height:
                print("TYPEII")
                new_height = 260
                new_width = int(260 * (width / height))

            haze_img = haze_img.resize((new_width, new_height), Image.ANTIALIAS)
            width, height = haze_img.size
            print(width, height)
        '''
        if width < crop_width or height < crop_height:
            if width < height:
                haze_img = haze_img.resize((260, int(260 * (height / width))), Image.ANTIALIAS)
                gt_img = gt_img.resize((260, int(260 * (height / width))), Image.ANTIALIAS)
            else:
                haze_img = haze_img.resize((int(260 * (width / height)), 260), Image.ANTIALIAS)
                gt_img = gt_img.resize((int(260 * (width / height)), 260), Image.ANTIALIAS)
            width, height = haze_img.size
        # --- x,y coordinate of left-top corner --- #
        if width - crop_width < 1:
            print(haze_img.size)
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        

        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)

        #if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
        #    raise Exception('Bad image channel: {}'.format(gt_name))

        #return haze, gt
        return haze, gt, haze_name


    def __len__(self):
        return len(self.haze_names)
