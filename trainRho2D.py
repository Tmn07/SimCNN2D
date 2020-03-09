# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:21:49 2019

@author: Alexis
"""

import torch
#import torch.optim as optim
import torch.nn as nn
#import torch.nn.functional as F
import numpy as np
from codec3060 import DenseED
#from models.elasticity import LossDirichlet,LossRho_norm4# LossNonlinear, NonLinearConstant  # LossEnu
# from models.darcy import *
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# from utils.image_gradient import SobelFilter
# from utils.load import load_data
from utils.misc import mkdirs, to_numpy
from utils.plot import plot_prediction_det, save_stats
from utils.practices import OneCycleScheduler, adjust_learning_rate#, find_lr
import time
import argparse
import random
from pprint import pprint
import json
# import sys
import matplotlib.pyplot as plt
# import h5py
# from FEA_simp import ComputeTarget
# from mytest import testsample, get_testsample

plt.switch_backend('agg')



def LoadData(device):
    Edata = np.loadtxt("data/SmapleLinear_30x60_bridge_X.txt", dtype=np.float32)
    data = Edata.reshape(-1, 1, 60, 30).transpose([0, 1, 3, 2])  # [bs, 1, 20, 40]
    print(f"{data.shape} input data loaded")
    
    data_u = np.loadtxt("data/SmapleLinear_30x60_bridge_U.txt", dtype=np.float32)
    ref_u0 = torch.from_numpy(data_u).unsqueeze(1).to(device)
    ref_uy = ref_u0[:, :, range(1, 3782, 2)]
    ref_ux = ref_u0[:, :, range(0, 3782, 2)]
    ref = torch.cat([ref_ux, ref_uy], 1).view(-1, 2, 61, 31).permute(0, 1, 3, 2)
    print(f"{ref.shape} ref data loaded")
    data = data[:64,]
    ref = ref[:64,]
    return data, ref

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Learning surrogate with mixed residual norm loss')
        self.add_argument('--exp-name', type=str, default='CNN', help='experiment name')
        self.add_argument('--exp-dir', type=str, default="./experiments", help='directory to save experiments')
        # codec
        self.add_argument('--blocks', type=list, default=[8, 12, 8],
                          help='list of number of layers in each dense block')
        self.add_argument('--growth-rate', type=int, default=16,
                          help='number of output feature maps of each conv layer within each dense block')
        self.add_argument('--init-features', type=int, default=48,
                          help='number of initial features after the first conv layer')
        self.add_argument('--drop-rate', type=float, default=0., help='dropout rate')
        self.add_argument('--upsample', type=str, default='nearest', choices=['nearest', 'bilinear'])
        # data
        self.add_argument('--data-dir', type=str, default="./datasets", help='directory to dataset')
        # self.add_argument('--data', type=str, default='grf_kle512', choices=['grf_kle512', 'channelized'])
        self.add_argument('--data', type=str, default='EnergyRho2D6030',
                          choices=['Enu', 'SIMP_penal1', 'SIMP_penal3_plus1'])
        # self.add_argument('--data', type=str, default='SIMP_penal1',
        # choices=['SIMP_penal1', 'grf_kle512', 'channelized'])
        self.add_argument('--ntrain', type=int, default=4096, help="number of training data")
        self.add_argument('--ntest', type=int, default=512, help="number of validation data")
        self.add_argument('--imsize', type=int, default=64)
        # training
        self.add_argument('--run', type=int, default=1, help='run instance')
        self.add_argument('--epochs', type=int, default=2000, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=1e-3, help='learnign rate')
        self.add_argument('--lr-div', type=float, default=2., help='lr div factor to get the initial lr')
        self.add_argument('--lr-pct', type=float, default=0.3,
                          help='percentage to reach the maximun lr, which is args.lr')
        self.add_argument('--weight-decay', type=float, default=0., help="weight decay")
        self.add_argument('--weight-bound', type=float, default=10, help="weight for boundary loss")
        self.add_argument('--batch-size', type=int, default=1, help='input batch size for training')
        self.add_argument('--test-batch-size', type=int, default=64, help='input batch size for testing')
        self.add_argument('--seed', type=int, default=1, help='manual seed used in Tensor')
        self.add_argument('--cuda', type=int, default=0, choices=[0, 1, 2, 3], help='cuda index')
        # logging
        self.add_argument('--debug', action='store_true', default=True, help='debug or verbose')
        self.add_argument('--ckpt-epoch', type=int, default=None, help='which epoch of checkpoints to be loaded')
        self.add_argument('--ckpt-freq', type=int, default=200, help='how many epochs to wait before saving model')
        self.add_argument('--log-freq', type=int, default=10,
                          help='how many epochs to wait before logging training status')
        self.add_argument('--plot-freq', type=int, default=50,
                          help='how many epochs to wait before plotting test output')
        self.add_argument('--plot-fn', type=str, default='imshow', choices=['contourf', 'imshow'],
                          help='plotting method')

    def parse(self):
        args = self.parse_args()
        tid = 2
        hparams = f'{args.data}_{tid}_run{args.run}_bs{args.batch_size}'
#        if args.debug:
#            hparams = 'debug/' + hparams
        args.run_dir = args.exp_dir + '/' + args.exp_name + '/' + hparams
        args.ckpt_dir = args.run_dir + '/checkpoints'
        # print(args.run_dir)
        # print(args.ckpt_dir)
        mkdirs(args.run_dir, args.ckpt_dir)

        # assert args.ntrain % args.batch_size == 0 and \
        #     args.ntest % args.test_batch_size == 0

        if args.seed is None:
            args.seed = random.randint(1, 10000)
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        print('Arguments:')
        pprint(vars(args))
        with open(args.run_dir + "/args.txt", 'w') as args_file:
            json.dump(vars(args), args_file, indent=4)

        return args

args = Parser().parse()
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
print(device)
args.train_dir = args.run_dir + '/training'
args.pred_dir = args.train_dir + '/predictions'
mkdirs(args.train_dir, args.pred_dir)


data, ref = LoadData(device)

model = DenseED(in_channels=1, out_channels=2,
                imsize=args.imsize,
                blocks=args.blocks,
                growth_rate=args.growth_rate,
                init_features=args.init_features,
                drop_rate=args.drop_rate,
                out_activation=None,
                upsample=args.upsample).to(device)
# modelname = "experiments/codec/mixed_residual/debug/SIMP_penal3_plus1_1_run1_bs64/checkpoints/model_epoch5300.pth"
# model = torch.load(modelname)

if args.debug:
    print(model)
    pass
# if start from ckpt
if args.ckpt_epoch is not None:
    ckpt_file = args.run_dir + f'/checkpoints/model_epoch{args.ckpt_epoch}.pth'
    model.load_state_dict(torch.load(ckpt_file, map_location='cpu'))
    print(f'Loaded ckpt: {ckpt_file}')
    print(f'Resume training from epoch {args.ckpt_epoch + 1} to {args.epochs}')

# G, B0, dNdx, dNdy, D, penal = NonLinearConstant(device)

data_tuple = (torch.FloatTensor(data).to(device), ref)
train_loader = DataLoader(TensorDataset(*data_tuple),
                          batch_size=args.batch_size, shuffle=True, drop_last=True)

# SGDM
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

# optimizer = optim.Adam(model.parameters(), lr=args.lr,
#                     weight_decay=args.weight_decay)
scheduler = OneCycleScheduler(lr_max=args.lr, div_factor=args.lr_div,
                              pct_start=args.lr_pct)

logger = {}
logger['loss_train'] = []
logger['loss_pde1'] = []
logger['loss_pde2'] = []
logger['loss_b'] = []
logger['u_l2loss'] = []
logger['ux_l2loss'] = []
logger['uy_l2loss'] = []
logger['s_l2loss'] = []

print('Start training...................................................')
start_epoch = 1 if args.ckpt_epoch is None else args.ckpt_epoch + 1
tic = time.time()
# step = 0
total_steps = args.epochs * len(train_loader)
print(f'total steps: {total_steps}')
for epoch in range(start_epoch, args.epochs + 1):
    model.train()

    relative_l2 = []
    relative_l1_ux = []
    relative_l1_uy = []
    loss_train, mse = 0., 0.
    loss_train_pde, loss_train_boundary = 0., 0.
    loss_train_pde_ux, loss_train_pde_uy = 0., 0.
    for batch_idx, (input, target) in enumerate(train_loader, start=1):
        input = input.to(device)
        model.zero_grad()
        output = model(input)

        # loss_pde1 = LossEnergy_free_penal1(input, output, device, Ke, Ff)
        loss_mse = F.mse_loss(output, target)
        loss = loss_mse
        loss.backward()

       # post-process for fixed boundary conditions
        output[:,:,30, 0] = 0
        output[:,1,30,60] = 0
        
        err2_sum = torch.sum((output - target) ** 2, [-1, -2])  # torch.Size([64, 2])
        relative_l2.append(torch.sqrt(err2_sum / torch.sum(target ** 2, [-1, -2])))

        def global_relative(output, target):
            output_x = output[:, :1]
            output_y = output[:, 1:]
            target_x = target[:, :1]
            target_y = target[:, 1:]
            relative_x = torch.sum((output_x - target_x) ** 2, [-1, -2]) / torch.sum(target_x ** 2, [-1, -2])
            relative_y = torch.sum((output_y - target_y) ** 2, [-1, -2]) / torch.sum(target_y ** 2, [-1, -2])
            return relative_x, relative_y

        # lr scheduling
        step = (epoch - 1) * len(train_loader) + batch_idx
        pct = step / total_steps
        lr = scheduler.step(pct)
        adjust_learning_rate(optimizer, lr)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        loss_train += loss.item()


    loss_train /= batch_idx

    rel2_cat = torch.cat(relative_l2, 0)  # torch.Size([1344, 2])
    re_l2 = to_numpy(torch.mean(rel2_cat, 0))
    relative_ux = re_l2[0]
    relative_uy = re_l2[1]

    print(
        f'Epoch {epoch}: training loss: {loss_train:.6f} ' \
        f'relative-ux: {relative_ux: .5f}, relative-uy: {relative_uy: .5f}')

    logf = open(args.train_dir + "/running_log.txt", 'a')
    logf.write(
        f'Epoch {epoch}: training loss: {loss_train:.6f}, ' \
        f'relative-ux: {relative_ux: .5f}, relative-uy: {relative_uy: .5f} \n')
    logf.close()



    if epoch % args.log_freq == 0:
        logger['loss_train'].append(loss_train)
        logger['ux_l2loss'].append(relative_ux)
        logger['uy_l2loss'].append(relative_uy)

    if epoch % args.ckpt_freq == 0:
        torch.save(model, args.ckpt_dir + "/model_epoch{}.pth".format(epoch))
        sampledir = args.ckpt_dir + "/model{}".format(epoch)
        mkdirs((sampledir))


tic2 = time.time()
print(f'Finished training {args.epochs} epochs with {args.ntrain} data ' \
      f'using {(tic2 - tic) / 60:.2f} mins')
metrics = ['loss_train', 'ux_l2loss', 'uy_l2loss']
save_stats(args.train_dir, logger, *metrics)
args.training_time = tic2 - tic
args.n_params, args.n_layers = model.model_size
with open(args.run_dir + "/args.txt", 'w') as args_file:
    json.dump(vars(args), args_file, indent=4)

