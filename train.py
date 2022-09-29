#!/usr/bin/env python
# -*- coding:utf-8 -*-
# IJCAI 2022
# Adaptive Convolutional Dictionary Network for CT Metal Artifact Reduction

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.functional as  F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
import numpy as np
from tensorboardX import SummaryWriter
from acdnet import ACDNet
from torch.utils.data import DataLoader
from Dataset import MARTrainDataset, MARValDataset
from math import ceil
from utils.utils import batch_PSNR
from utils.SSIM import SSIM
from utils import utils_image
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/train/", help='txt path to training data')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--patchSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=300, help='total number of training epochs')
parser.add_argument('--batchnum', type=int, default=800, help='the number of batch')
parser.add_argument('--N', type=int, default=6, help='the number of feature maps')
parser.add_argument('--Np', type=int, default=32, help='the number of channel concatenation')
parser.add_argument('--d', type=int, default=32, help='the number of convolutional filters in common dictionary D')
parser.add_argument('--num_res', type=int, default=3, help='Resblocks number in each ResNet')
parser.add_argument('--T', type=int, default=10, help='Stage number T')
parser.add_argument('--Mtau', type=float, default=0.5, help='for sparse feature map')
parser.add_argument('--etaM', type=float, default=1, help='stepsize for updating M')
parser.add_argument('--etaX', type=float, default=5, help='stepsize for updating X')
parser.add_argument('--resume', type=int, default=0, help='continue to train from epoch')
parser.add_argument("--milestone", type=int, default=[50,100,150,200], nargs = '+',
                    help="When to decay learning rate")
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument('--log_dir', default='logs/', help='tensorboard logs')
parser.add_argument('--model_dir', default='models/', help='saving model')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--Xl2', type=float, default=1, help='loss weights')
parser.add_argument('--Xl1', type=float, default=5e-4, help='loss weights')
parser.add_argument('--Al1', type=float, default=5e-4, help='loss weights')
opt = parser.parse_args()
if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

_modes = ['train', 'val']
metric = SSIM()
# create path
try:
    os.makedirs(opt.log_dir)
except OSError:
    pass
try:
    os.makedirs(opt.model_dir)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

def train_model(net, optimizer, lr_scheduler, datasets):
    batch_size = {'train': opt.batchSize, 'val': 1}
    data_loader = {phase: DataLoader(datasets[phase], batch_size=batch_size[phase], shuffle=True,
                                     num_workers=int(opt.workers), pin_memory=True) for phase in _modes}
    num_data = {phase: len(datasets[phase]) for phase in _modes}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in _modes}
    writer = SummaryWriter(opt.log_dir)
    step = 0
    for epoch in range(opt.resume, opt.niter):
        mse_per_epoch = {x: 0 for x in _modes}
        tic = time.time()
        # train stage
        lr = optimizer.param_groups[0]['lr']
        phase = 'train'
        optimizer.zero_grad()
        for ii, data in enumerate(data_loader[phase]):
            Xma, GT, XLI, mask = [x.cuda() for x in data]
            net.train()
            optimizer.zero_grad()
            X0, ListX, ListA, ListX_nonK, ListA_nonK= net(Xma, XLI, mask)
            loss_l2Xt = 0
            loss_l1Xt = 0
            loss_l1At = 0
            newAgt = mask*(Xma - GT)
            newXgt = mask * GT
            for j in range(opt.T):
                loss_l2Xt = loss_l2Xt +  0.1 * F.mse_loss(ListX[j]*mask, newXgt)
                loss_l1Xt = loss_l1Xt +  0.1 * torch.sum(torch.abs(ListX[j]*mask - newXgt))
                loss_l1At = loss_l1At +  0.1 * torch.sum(torch.abs(mask *ListA[j]-newAgt))
            loss_l1Xf = torch.sum(torch.abs((ListX[-1]*mask - newXgt)))
            loss_l1Af = torch.sum(torch.abs(mask *ListA[-1]-newAgt))
            loss_l2Xf = F.mse_loss(ListX[-1]*mask,newXgt)
            loss_l1X0=  0.1 * torch.sum(torch.abs(X0 * mask - newXgt))
            loss_l2X0 = 0.1 * F.mse_loss(X0 * mask, newXgt)
            loss_l2Xrcd = 0.1 * F.mse_loss(ListX_nonK[0]*mask, newXgt) + 1 * F.mse_loss(ListX_nonK[1]*mask, newXgt)
            loss_l1Xrcd= 0.1 * torch.sum(torch.abs((ListX_nonK[0]*mask-newXgt))) + 1 * torch.sum(torch.abs((ListX_nonK[1]*mask-newXgt)))
            loss_l1Arcd = 0.1 * torch.sum(torch.abs(mask *ListA_nonK[0]-newAgt)) + 1 * torch.sum(torch.abs((mask *ListA_nonK[1]-newAgt)))
            loss_l2X = loss_l2Xt + loss_l2Xf + loss_l2X0 + loss_l2Xrcd
            loss_l1X = loss_l1Xt + loss_l1Xf + loss_l1X0 + loss_l1Xrcd
            loss_l1A =  loss_l1At + loss_l1Af + loss_l1Arcd
            loss = opt.Xl2 * loss_l2X + opt.Xl1* loss_l1X + opt.Al1 * loss_l1A
            # back propagation
            loss.backward()
            optimizer.step()
            mse_iter = loss.item()
            mse_per_epoch[phase] += mse_iter
            Xoutclip = torch.clamp(ListX[-1]/255.0,0,0.5)
            Xgtclip = torch.clamp(GT/255.0, 0,0.5)
            rmseu = torch.sqrt(torch.mean(((Xoutclip - Xgtclip) * mask) ** 2))
            train_psnr = batch_PSNR(Xoutclip * mask, Xgtclip* mask, 0.5)
            train_ssim = metric(Xoutclip / 0.5 * mask, Xgtclip/0.5  * mask)
            if ii % 100 == 0:
                template = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>5d}/{:0>5d}, Loss={:5.2e}, Lossl2X={:5.2e},  Lossl1X={:5.2e}, Lossl1A={:5.2e}, lr={:.2e}'
                print(template.format(epoch + 1, opt.niter, phase, ii, num_iter_epoch[phase], mse_iter, loss_l2X, loss_l1X, loss_l1A, lr))
                log_str = 'rmseu={:5.4f},psnr={:4.2f}, ssim= {:5.4f}'
                print(log_str.format(rmseu.item(),train_psnr.item(), train_ssim.item()))
            writer.add_scalar('Train Loss Iter', mse_iter, step)
            step += 1
        mse_per_epoch[phase] /= (ii + 1)
        print('{:s}: Loss={:+.2e}'.format(phase, mse_per_epoch[phase]))
        print('-' * 100)
        # evaluation stage
        net.eval()
        psnr_per_epoch = 0
        ssim_per_epoch = 0
        phase = 'val'
        for ii, data in enumerate(data_loader[phase]):
            Xma, GT, XLI, mask = [x.cuda() for x in data]
            with torch.set_grad_enabled(False):
                X0, ListX, ListA, ListX_nonK, ListA_nonK= net(Xma, XLI, mask)
            newXgt = mask * GT
            Xoutclip = torch.clamp(ListX[-1]/255.0,0,0.5)
            Xgtclip = torch.clamp(GT/255.0,0,0.5)
            Out_img = utils_image.tensor2uint(Xoutclip/0.5)
            B_img = utils_image.tensor2uint(Xgtclip/0.5)
            psnr_iter = utils_image.calculate_psnr(Out_img * mask.data.squeeze().float().cpu().numpy(),
                                                      B_img * mask.data.squeeze().float().cpu().numpy())
            psnr_per_epoch += psnr_iter
            ssim_iter = utils_image.calculate_ssim(Out_img * mask.data.squeeze().float().cpu().numpy(),
                                                      B_img * mask.data.squeeze().float().cpu().numpy())
            ssim_per_epoch += ssim_iter
            ListX[-1].clamp_(0.0, 255.0)
            mse_iter = F.mse_loss(mask *ListX[-1], newXgt)
            mse_per_epoch[phase] += mse_iter
            if ii % 100 == 0:
                log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>3d}/{:0>3d}, mae={:.2e}, psnr={:4.2f}, ssim= {:5.4f}'
                print(log_str.format(epoch + 1, opt.niter, phase, ii + 1, num_iter_epoch[phase], mse_iter, psnr_iter, ssim_iter))
        psnr_per_epoch /= (ii + 1)
        ssim_per_epoch /= (ii + 1)
        mse_per_epoch[phase] /= (ii + 1)
        print('{:s}: mse={:.3e}, PSNR={:4.2f}, ssim= {:5.4f}'.format(phase, mse_per_epoch[phase], psnr_per_epoch, ssim_per_epoch))
        print('-' * 100)
        # adjust the learning rate
        lr_scheduler.step()
        # save model
        torch.save(net.state_dict(), os.path.join(opt.model_dir, 'ACDNet_latest.pt'))
        if (epoch+1) % 20 == 0:
            # save model
            model_prefix = 'model_'
            save_path_model = os.path.join(opt.model_dir, model_prefix + str(epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'step': step + 1,
            }, save_path_model)
            torch.save(net.state_dict(), os.path.join(opt.model_dir, 'ACDNet_state_%d.pt' % (epoch+1)))
        writer.add_scalars('MSE_epoch', mse_per_epoch, epoch + 1)
        writer.add_scalar('val PSNR epoch', psnr_per_epoch, epoch + 1)
        writer.add_scalar('val SSIM epoch', ssim_per_epoch, epoch + 1)
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc - tic))
    writer.close()
    print('Reach the maximal epochs! Finish training')

if __name__ == '__main__':
    def print_network(net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('Total number of parameters: %d' % num_params)
    netACD = ACDNet(opt).cuda()
    print_network(netACD)
    optimizerACD = optim.Adam(netACD.parameters(), betas=(0.5, 0.999), lr=opt.lr)
    schedulerACD = optim.lr_scheduler.MultiStepLR(optimizerACD, milestones=opt.milestone, gamma=0.5)
    # from opt.resume continue to train
    for _ in range(opt.resume):
        schedulerACD.step()
    if opt.resume:
        checkpoint = torch.load(os.path.join(opt.model_dir, 'model_' + str(opt.resume)))
        netACD.load_state_dict(torch.load(os.path.join(opt.model_dir, 'ACDNet_state_' + str(opt.resume) + '.pt')))
        print('loaded checkpoints, epoch{:d}'.format(checkpoint['epoch']))
    # load dataset
    train_mask = np.load(os.path.join(opt.data_path, 'trainmask.npy'))
    train_dataset = MARTrainDataset(opt.data_path, opt.patchSize, int(opt.batchSize * opt.batchnum), train_mask)
    val_dataset = MARValDataset(opt.data_path, train_mask)
    datasets = {'train': train_dataset, 'val': val_dataset}
    # train model
    train_model(netACD, optimizerACD, schedulerACD, datasets)
