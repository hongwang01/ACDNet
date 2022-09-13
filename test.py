import os
import os.path
import argparse
import numpy as np
import torch
import time
import h5py
from utils import utils_image
import PIL
from PIL import Image
import utils.save_image as save_img
from acdnet import ACDNet

parser = argparse.ArgumentParser(description="ACDNet_Test")
parser.add_argument("--model_dir", type=str, default="models/ACDNet_latest.pt", help='path to model file')
parser.add_argument("--data_path", type=str, default="data/test/", help='path to test data')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--save_path", type=str, default="save_results/", help='path to testing results')
parser.add_argument('--N', type=int, default=6, help='the number of feature maps')
parser.add_argument('--Np', type=int, default=32, help='the number of channel concatenation')
parser.add_argument('--d', type=int, default=32, help='the number of convolutional filters in common dictionary D')
parser.add_argument('--num_res', type=int, default=3, help='Resblocks number in each ResNet')
parser.add_argument('--T', type=int, default=10, help='Stage number T')
parser.add_argument('--Mtau', type=float, default=1.5, help='for sparse feature map')
parser.add_argument('--etaM', type=float, default=1, help='stepsize for updating M')
parser.add_argument('--etaX', type=float, default=5, help='stepsize for updating X')
parser.add_argument('--batchSize', type=int, default=1, help='testing input batch size')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")

out_dir = opt.save_path+'/ACDNet/image/'
out_hudir = opt.save_path+'/ACDNet/hu/'
mkdir(out_dir)
mkdir(out_hudir)

input_dir = opt.save_path+'/input/image/'
input_hudir = opt.save_path+'/input/hu/'
mkdir(input_dir)
mkdir(input_hudir)

gt_dir = opt.save_path+'/gt/image/'
gt_hudir = opt.save_path+'/gt/hu/'
mkdir(gt_dir)
mkdir(gt_hudir)

def normalized(X):
    maxX = np.max(X)
    minX = np.min(X)
    X = (X - minX) / (maxX - minX)
    return X

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

def image_get_minmax():
    return 0.0, 1.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data

def tohu(X):           # display window as [-175HU, 275HU]
    CT = (X - 0.192) * 1000 / 0.192
    CT_win = CT.clamp_(-175, 275)
    CT_winnorm = (CT_win +175) / (275+175)
    return CT_winnorm

test_mask = np.load(os.path.join(opt.data_path, 'testmask.npy'))
def test_image(data_path, imag_idx, mask_idx):
    txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    mat_files = open(txtdir, 'r').readlines()
    gt_dir = mat_files[imag_idx]
    file_dir = gt_dir[:-6]
    data_file = file_dir + str(mask_idx) + '.h5'
    abs_dir = os.path.join(data_path, 'test_640geo/', data_file)
    gt_absdir = os.path.join(data_path, 'test_640geo/', gt_dir[:-1])
    gt_file = h5py.File(gt_absdir, 'r')
    Xgt = gt_file['image'][()]
    gt_file.close()
    file = h5py.File(abs_dir, 'r')
    Xma= file['ma_CT'][()]
    XLI =file['LI_CT'][()]
    file.close()
    M512 = test_mask[:,:,mask_idx]
    M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
    Xma = normalize(Xma, image_get_minmax())  
    Xgt = normalize(Xgt, image_get_minmax())
    XLI = normalize(XLI, image_get_minmax())
    Mask = M.astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)),0)
    non_mask = 1 - Mask
    return torch.Tensor(Xma).cuda(), torch.Tensor(Xgt).cuda(), torch.Tensor(XLI).cuda(), torch.Tensor(non_mask).cuda()
def main():
    # Build model
    print('Loading model ...\n')
    model = ACDNet(opt).cuda()
    model.load_state_dict(torch.load(opt.model_dir))
    model.eval()
    time_test = 0
    count = 0
    #for imag_idx in range(200): # for original testing, 200 clean CT images
    for imag_idx in range(1):  # for demo testing, we only provide one testing data as "test_640geo/000376_02_01/040/0.h5" due to the limitation of file size about supplementary material
        print("imag_idx:",imag_idx)
      #  for mask_idx in range(10): # for original testing, 10 testing metal masks
        for mask_idx in range(1):   # for demo testing, we only 1 testing metal masks
            Xma, X, XLI, M = test_image(opt.data_path, imag_idx, mask_idx)
            with torch.no_grad():
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()
                X0, ListX, ListA, ListX_nonK, ListA_nonK = model(Xma, XLI, M)
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time
            Xoutclip = torch.clamp(ListX[-1] / 255.0, 0, 0.5)
            Xgtclip = torch.clamp(X / 255.0, 0, 0.5)
            Xmaclip = torch.clamp(Xma / 255.0, 0, 0.5)
            Xoutnorm = Xoutclip / 0.5
            Xgtnorm = Xgtclip / 0.5
            Xmanorm = Xmaclip / 0.5
            Xouthu = tohu(Xoutclip)
            Xgthu = tohu(Xgtclip)
            Xmahu = tohu(Xmaclip)
            idx = imag_idx *10+ mask_idx  + 1
            Xnorm = [Xoutnorm, Xmanorm, Xgtnorm]
            Xhu = [Xouthu, Xmahu, Xgthu]
            dir = [out_dir, input_dir, gt_dir]
            hudir = [out_hudir, input_hudir, gt_hudir]
            save_img.imwrite(idx, dir, Xnorm)
            save_img.imwrite(idx, hudir,Xhu)
            print('Times: ', dur_time)
            count += 1
    print(100*'*')
if __name__ == "__main__":
    main()

