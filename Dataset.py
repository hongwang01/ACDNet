import os
import os.path
import numpy as np
import random
import h5py
import torch
import torch.utils.data as udata
from numpy.random import RandomState
import PIL
from PIL import Image


def image_get_minmax():
    return 0.0, 1.0


def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data * 2.0 - 1.0
    data = data.astype(np.float32)
    data = np.transpose(np.expand_dims(data, 2), (2, 0, 1))
    return data


def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    def _augment(img):
        if hflip: img = img[:, ::-1]
        if vflip: img = img[::-1, :]
        return img
    return [_augment(a) for a in args]

class MARTrainDataset(udata.Dataset):
    def __init__(self, dir, patchSize, length, mask):
        super().__init__()
        self.dir = dir
        self.train_mask = mask
        self.patch_size = patchSize
        self.sample_num = length
        self.txtdir = os.path.join(self.dir, 'train_640geo_dir.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.rand_state = RandomState(66)
        self.start=0
        self.end = int(self.file_num * 0.9)     # 90% of training data
    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        gt_dir = self.mat_files[idx % self.end]
        #random_mask = random.randint(0, 79) # for original training, the total number of training metal masks is 80
        random_mask = 0  # for demo training,  we only provide one training data as "train_640geo/000195_02_02/202/0.h5" due to the limitation of file size about supplementary material
        file_dir = gt_dir[:-6]
        data_file = file_dir + str(random_mask) + '.h5'
        abs_dir = os.path.join(self.dir, 'train_640geo/', data_file)
        gt_absdir = os.path.join(self.dir,'train_640geo/', gt_dir[:-1])
        gt_file = h5py.File(gt_absdir, 'r')
        Xgt = gt_file['image'][()]
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        Xma= file['ma_CT'][()]
        XLI =file['LI_CT'][()]
        file.close()
        M512 = self.train_mask[:,:,random_mask]
        M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
        Xgtclip = np.clip(Xgt,0,1)
        Xgtnorm = Xgtclip
        Xmaclip = np.clip(Xma, 0, 1)
        Xmanorm = Xmaclip
        XLIclip = np.clip(XLI, 0,1)
        XLInorm = XLIclip
        O = Xmanorm*255
        O, row, col = self.crop(O)
        B = Xgtnorm*255
        B = B[row: row + self.patch_size, col: col + self.patch_size]
        LI = XLInorm*255
        LI = LI[row: row + self.patch_size, col: col + self.patch_size]
        M = M[row: row + self.patch_size, col: col + self.patch_size]
        O = O.astype(np.float32)
        LI = LI.astype(np.float32)
        B = B.astype(np.float32)
        Mask = M.astype(np.float32)
        O, B, LI, Mask = augment(O, B, LI, Mask)
        O = np.transpose(np.expand_dims(O, 2), (2, 0, 1))
        B = np.transpose(np.expand_dims(B, 2), (2, 0, 1))
        LI = np.transpose(np.expand_dims(LI, 2), (2, 0, 1))
        Mask = np.transpose(np.expand_dims(Mask, 2), (2, 0, 1))
        non_Mask = 1 - Mask #non_metal region
        return torch.from_numpy(O.copy()),torch.from_numpy(B.copy()),torch.from_numpy(LI.copy()),torch.from_numpy(non_Mask.copy())

    def crop(self, img):
        h, w = img.shape
        p_h, p_w = self.patch_size, self.patch_size
        if h == p_h:
            r = 0
            c = 0
            O = img
        else:
            r = self.rand_state.randint(0, h - p_h)
            c = self.rand_state.randint(0, w - p_w)
            O = img[r: r + p_h, c: c + p_w]
        return O, r, c

class MARValDataset(udata.Dataset):
    def __init__(self, dir, mask):
        super().__init__()
        self.dir = dir
        self.train_mask = mask
        self.txtdir = os.path.join(self.dir, 'train_640geo_dir.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.rand_state = RandomState(66)
        self.start = int(self.file_num * 0.9)
        self.end = int(self.file_num)
        self.sample_num = self.end - self.start  ## 10% of training data as validation data
    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        gt_dir = self.mat_files[idx+self.start]
     #   random_mask = random.randint(80,89)      # for original validation, the total number of metal masks for validation is 10
        random_mask = 80   # for demo validation,  we only provide one validation data as "train_640geo/000129_03_01/115/80.h5" due to the limitation of file size about supplementary material

        file_dir = gt_dir[:-6]
        data_file = file_dir + str(random_mask) + '.h5'
        abs_dir = os.path.join(self.dir, 'train_640geo/', data_file)
        gt_absdir = os.path.join(self.dir,'train_640geo/', gt_dir[:-1])
        gt_file = h5py.File(gt_absdir, 'r')
        Xgt = gt_file['image'][()]
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        Xma= file['ma_CT'][()]
        XLI =file['LI_CT'][()]
        file.close()
       # M512 = self.train_mask[:, :, random_mask] # for original training
        M512 = self.train_mask[:, :, 1] # for demo training
        M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
        Xgtclip = np.clip(Xgt,0,1)
        Xgtnorm = Xgtclip
        Xmaclip = np.clip(Xma, 0, 1)
        Xmanorm = Xmaclip
        XLIclip = np.clip(XLI, 0,1)
        XLInorm = XLIclip
        O = Xmanorm*255
        B = Xgtnorm*255
        LI = XLInorm*255
        O = O.astype(np.float32)
        LI = LI.astype(np.float32)
        B = B.astype(np.float32)
        Mask = M.astype(np.float32)
        O = np.transpose(np.expand_dims(O, 2), (2, 0, 1))
        B = np.transpose(np.expand_dims(B, 2), (2, 0, 1))
        LI = np.transpose(np.expand_dims(LI, 2), (2, 0, 1))
        Mask = np.transpose(np.expand_dims(Mask, 2), (2, 0, 1))
        non_Mask = 1 - Mask
        return torch.from_numpy(O.copy()),torch.from_numpy(B.copy()),torch.from_numpy(LI.copy()),torch.from_numpy(non_Mask.copy())

class TestDataset(udata.Dataset):
    def __init__(self, dir, mask):
        super().__init__()
        self.dir = dir
        self.test_mask = mask
        self.txtdir = os.path.join(self.dir, 'test_640geo_dir.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.sample_num = self.file_num * 10   # every clean CT image for testing is paired with 10 metal masks.
        self.rand_state = RandomState(66)

    def __len__(self):
        return int(self.sample_num)

    def __getitem__(self, idx):
        gt_dir = self.mat_files[idx % int(self.file_num)]
        random_mask = random.randint(0, 9)
        file_dir = gt_dir[:-6]
        data_file = file_dir + str(random_mask) + '.h5'
        abs_dir = os.path.join(self.dir, 'test_640geo/', data_file)
        gt_absdir = os.path.join(self.dir, 'test_640geo/', gt_dir[:-1])
        gt_file = h5py.File(gt_absdir, 'r')
        Xgt = gt_file['image'][()]
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        Xma = file['ma_CT'][()]
        XLI = file['LI_CT'][()]
        file.close()
        M512 = self.test_mask[:, :, random_mask]
        M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
        Xgtclip = np.clip(Xgt, 0, 1)
        Xgtnorm = Xgtclip
        Xmaclip = np.clip(Xma, 0, 1)
        Xmanorm = Xmaclip
        XLIclip = np.clip(XLI, 0, 1)
        XLInorm = XLIclip
        O = Xmanorm * 255
        O = O.astype(np.float32)
        O = np.transpose(np.expand_dims(O, 2), (2, 0, 1))
        B = Xgtnorm * 255
        B = B.astype(np.float32)
        B = np.transpose(np.expand_dims(B, 2), (2, 0, 1))
        LI = XLInorm * 255
        LI = LI.astype(np.float32)
        LI = np.transpose(np.expand_dims(LI, 2), (2, 0, 1))
        Mask = M.astype(np.float32)
        Mask = np.transpose(np.expand_dims(Mask, 2), (2, 0, 1))
        non_Mask = 1 - Mask
        return torch.Tensor(O), torch.Tensor(B), torch.Tensor(LI), torch.Tensor(non_Mask)
