import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import lpips


lpips_fn = lpips.LPIPS(net='alex')  # 只初始化一次

def compute_measure(x, y, pred, data_range):
    original_psnr = compute_PSNR(x, y, data_range)
    original_ssim = compute_SSIM(x, y, data_range)
    original_rmse = compute_RMSE(x, y)
    pred_psnr = compute_PSNR(pred, y, data_range)
    pred_ssim = compute_SSIM(pred, y, data_range)
    pred_rmse = compute_RMSE(pred, y)
    
    lpips_pred = lpips_fn(to_lpips_tensor(pred), to_lpips_tensor(y)).item()
    lpips_ori = lpips_fn(to_lpips_tensor(x), to_lpips_tensor(y)).item()
    
    return (original_psnr, original_ssim, original_rmse, lpips_ori), (pred_psnr, pred_ssim, pred_rmse, lpips_pred)

def to_lpips_tensor(img):
    if isinstance(img, torch.Tensor):
        if img.dim() == 2:
            img = img.unsqueeze(0)
        if img.size(0) == 1:
            img = img.unsqueeze(0)
        if img.size(1) == 1:
            img = img.repeat(1,3,1,1)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 2 - 1
        return img
    else:
        raise ValueError("LPIPS must be torch.Tensor")

def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()


def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2, data_range):
    if type(img1) == torch.Tensor:
        mse_ = compute_MSE(img1, img2)
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        mse_ = compute_MSE(img1, img2)
        return 10 * np.log10((data_range ** 2) / mse_)


def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
    # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
    if len(img1.size()) == 2:
        shape_ = img1.shape[-1]
        img1 = img1.view(1,1,shape_ ,shape_ )
        img2 = img2.view(1,1,shape_ ,shape_ )
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    #C1, C2 = 0.01**2, 0.03**2

    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
