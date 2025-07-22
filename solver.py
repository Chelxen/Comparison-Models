import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from prep import printProgressBar
from measure import compute_measure

from REDCNN.networks import RED_CNN
from MambaIR.mambair_arch import MambaIR
from DeepGuess.architectures import ResUNet
from UKAN.archs import UKAN
from DenoMamba.model.denomamba_arch import DenoMamba

class Solver(object):
    def __init__(self, args, data_loader, model_name):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader
        self.model_name = model_name

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size

        # Modified the model to other comparison Models here
        # self.REDCNN = RED_CNN()
        if args.model_name == 'REDCNN':
            self.REDCNN = RED_CNN()
        elif args.model_name == 'MambaIR':
            self.REDCNN = MambaIR(img_size = self.patch_size, patch_size=8, in_chans=1, embed_dim=96, upscale=1, img_range=1., upsampler='')
        elif args.model_name == 'DeepGuess':
            self.REDCNN = ResUNet(img_ch=1, output_ch=1)
        elif args.model_name == 'DenoMamba':
            self.REDCNN = DenoMamba(inp_channels=1, out_channels=1)
        elif args.model_name == 'UKAN':
            self.REDCNN = UKAN(num_classes=1, input_channels=1, img_size=self.patch_size, patch_size=8, embed_dims=[256, 320, 512])

        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.REDCNN = nn.DataParallel(self.REDCNN)
        self.REDCNN.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.REDCNN.parameters(), self.lr)


    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        torch.save(self.REDCNN.state_dict(), f)
        # 保存优化器状态
        f_opt = os.path.join(self.save_path, 'optimizer_{}iter.pth'.format(iter_))
        torch.save(self.optimizer.state_dict(), f_opt)


    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.REDCNN.load_state_dict(state_d)
        else:
            self.REDCNN.load_state_dict(torch.load(f))
        # 加载优化器状态
        f_opt = os.path.join(self.save_path, 'optimizer_{}iter.pth'.format(iter_))
        if os.path.exists(f_opt):
            self.optimizer.load_state_dict(torch.load(f_opt))


    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()


    def train(self):
        train_losses = []
        total_iters = 0
        start_time = time.time()
        # resume training from checkpoint
        if hasattr(self, 'args') and getattr(self, 'args', None) is not None:
            args = self.args
        else:
            import sys
            args = sys.modules['__main__'].args if hasattr(sys.modules['__main__'], 'args') else None
        if args and getattr(args, 'resume', False) and getattr(args, 'resume_iters', 0) > 0:
            self.load_model(args.resume_iters)
            total_iters = args.resume_iters
            # also load loss file if it exists
            loss_file = os.path.join(self.save_path, 'loss_{}_iter.npy'.format(args.resume_iters))
            if os.path.exists(loss_file):
                train_losses = list(np.load(loss_file))

        for epoch in range(1, self.num_epochs):
            self.REDCNN.train(True)

            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1

                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                if self.patch_size: # patch training
                    x = x.view(-1, 1, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)

                pred = self.REDCNN(x)
                loss = self.criterion(pred, y)
                self.REDCNN.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        self.num_epochs, iter_+1, 
                                                                                                        len(self.data_loader), loss.item(), 
                                                                                                        time.time() - start_time))
                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model
                if total_iters % self.save_iters == 0:
                    self.save_model(total_iters)
                    np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))


    def test(self):
        del self.REDCNN
        # load
        if self.model_name == 'REDCNN':
            self.REDCNN = RED_CNN()
        elif self.model_name == 'MambaIR':
            self.REDCNN = MambaIR(img_size = self.patch_size, patch_size=8, in_chans=1, embed_dim=96, upscale=1, img_range=1., upsampler='')
        elif self.model_name == 'DeepGuess':
            self.REDCNN = ResUNet(img_ch=1, output_ch=1)
        elif self.model_name == 'DenoMamba':
            self.REDCNN = DenoMamba(inp_channels=1, out_channels=1)
        elif self.model_name == 'UKAN':
            self.REDCNN = UKAN(num_classes=1, input_channels=1, img_size=self.patch_size, patch_size=8, embed_dims=[256, 320, 512])
        self.REDCNN.to(self.device)
        self.load_model(self.test_iters)

        # compute PSNR, SSIM, RMSE, LPIPS
        ori_psnr_list, ori_ssim_list, ori_rmse_list, ori_lpips_list = [], [], [], []
        pred_psnr_list, pred_ssim_list, pred_rmse_list, pred_lpips_list = [], [], [], []

        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                pred = self.REDCNN(x)

                # denormalize, truncate
                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_list.append(original_result[0])
                ori_ssim_list.append(original_result[1])
                ori_rmse_list.append(original_result[2])
                ori_lpips_list.append(original_result[3])
                pred_psnr_list.append(pred_result[0])
                pred_ssim_list.append(pred_result[1])
                pred_rmse_list.append(pred_result[2])
                pred_lpips_list.append(pred_result[3])

                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result)

                printProgressBar(i, len(self.data_loader),
                                 prefix="Compute measurements ..",
                                 suffix='Complete', length=25)
            print('\n')

            def mean_std_str(arr):
                arr = np.array(arr)
                return "{:.4f}±{:.4f}".format(arr.mean(), arr.std())
            
            ori_psnr_str = mean_std_str(ori_psnr_list)
            ori_ssim_str = mean_std_str(ori_ssim_list)
            ori_rmse_str = mean_std_str(ori_rmse_list)
            ori_lpips_str = mean_std_str(ori_lpips_list)

            pred_psnr_str = mean_std_str(pred_psnr_list)
            pred_ssim_str = mean_std_str(pred_ssim_list)
            pred_rmse_str = mean_std_str(pred_rmse_list)
            pred_lpips_str = mean_std_str(pred_lpips_list)

            result_str = ''
            result_str += 'Original ===\nPSNR: {}\nSSIM: {}\nRMSE: {}\nLPIPS: {}\n'.format(ori_psnr_str, ori_ssim_str, ori_rmse_str, ori_lpips_str)
            result_str += '\nPredictions ===\nPSNR: {}\nSSIM: {}\nRMSE: {}\nLPIPS: {}\n'.format(pred_psnr_str, pred_ssim_str, pred_rmse_str, pred_lpips_str)
            print(result_str)

            # 保存到txt
            with open(os.path.join(self.save_path, 'test_results.txt'), 'w') as f:
                f.write(result_str)
