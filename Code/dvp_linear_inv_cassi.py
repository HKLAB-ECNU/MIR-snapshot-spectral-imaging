import time
import math
import numpy as np
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
# from skimage.measure import (compare_psnr, compare_ssim)
from utils import (A, At, psnr, shift, shift_back,calculate_ssim,TV_denoiser)
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics  import structural_similarity as compare_ssim
from hsi import HSI_SDeCNN as net
import torch
from bm3d import bm3d_deblurring, BM3DProfile, gaussian_kernel
import scipy.io as sio

def gap_denoise(y, Phi, A, At, _lambda=1, accelerate=True, 
                denoiser='tv', iter_max=50, noise_estimate=True, sigma=None, 
                tv_weight=0.1, tv_iter_max=5, multichannel=True, x0=None,
                X_orig=None, model=None, show_iqa=True):
    if x0 is None:
        print(At)
        x0 = At(y, Phi) # default start point (initialized value)
    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(iter_max, list):
        iter_max = [iter_max] * len(sigma)
    y1 = np.zeros_like(y) 
    Phi_sum = np.sum(Phi,2)
    Phi_sum[Phi_sum==0]=1
    # [1] start iteration for reconstruction
    x = x0 # initialization
    psnr_all = []
    ssim_all=[]
    k = 0
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = net()
    model.load_state_dict(torch.load(r'./check_points/deep_denoiser.pth',weights_only=True))
    model.eval()
    for q, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    for idx, nsig in enumerate(sigma): # iterate all noise levels
        for it in range(iter_max[idx]):
            #print('max1_{0}_{1}:'.format(idx,it),np.max(x))
            yb = A(x,Phi)
            if accelerate: # accelerated version of GAP
                y1 = y1 + (y-yb)
                x = x + _lambda*(At((y1-yb)/Phi_sum,Phi)) # GAP_acc
            else:
                x = x + _lambda*(At((y-yb)/Phi_sum,Phi)) # GAP
            # x = shift_back(x,step=1)
            # switch denoiser
            if denoiser.lower() == 'tv': # total variation (TV) denoising
                x = denoise_tv_chambolle(x, nsig / 255, max_num_iter= tv_iter_max ,channel_axis=-1 if multichannel else None)
                #x= TV_denoiser(x, tv_weight, n_iter_max=tv_iter_max)
            elif denoiser.lower() == 'hsicnn':
                l_ch=10
                m_ch=10
                h_ch=10
                if (k>123 and k<=125 ) or (k>=119 and k<=121) or (k>=115 and k<=117) or (k>=111 and k<=113) or (k>=107 and k<=109) or (k>=103 and k<=105) or (k>=99 and k<=101) or (k>=95 and k<=97) or  (k>=91 and k<=93) or (k>=87 and k<=89) or (k>=83 and k<=85):
                    tem = None
                    for i in range(31):
                        net_input = None

                        if i < 3:
                            ori_nsig = nsig

                            if i==0:
                                net_input = np.dstack((x[:, :, i], x[:, :, i], x[:, :, i], x[:, :, i:i + 4]))
                            elif i==1:
                                net_input = np.dstack((x[:, :, i-1], x[:, :, i-1], x[:, :, i-1], x[:, :, i:i + 4]))
                            elif i==2:
                                net_input = np.dstack((x[:, :, i-2], x[:, :, i-2], x[:, :, i-1], x[:, :, i:i + 4]))
                            net_input = torch.from_numpy(np.ascontiguousarray(net_input)).permute(2,0,1).float().unsqueeze(0)
                            net_input = net_input.to(device)
                            Nsigma = torch.full((1, 1, 1, 1), l_ch / 255.).type_as(net_input)
                            output = model(net_input, Nsigma)
                            output = output.data.squeeze().cpu().numpy()
                            if k<0:
                                output = denoise_tv_chambolle(x[:, :, i], nsig / 255, max_num_iter=tv_iter_max,multichannel=False)
                            nsig = ori_nsig
                            if i == 0:
                                tem = output
                            else:
                                tem = np.dstack((tem, output))
                        elif i > 27:
                            ori_nsig=nsig
                            if k>=45:
                                nsig/=1
                            if i==28:
                                net_input = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i+1], x[:, :, i+2], x[:, :, i+2]))
                            elif i==29:
                                net_input = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i+1], x[:, :, i+1], x[:, :, i+1]))
                            elif i==30:
                                net_input = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i], x[:, :, i], x[:, :, i]))
                            net_input = torch.from_numpy(np.ascontiguousarray(net_input)).permute(2, 0,1).float().unsqueeze(0)
                            net_input = net_input.to(device)
                            Nsigma = torch.full((1, 1, 1, 1), m_ch / 255.).type_as(net_input)
                            output = model(net_input, Nsigma)
                            output = output.data.squeeze().cpu().numpy()
                            if k<0:
                                output = denoise_tv_chambolle(x[:, :, i], 10 / 255, max_num_iter=tv_iter_max,multichannel=False)
                            tem = np.dstack((tem, output))
                            nsig=ori_nsig

                        else:
                            ori_nsig = nsig
                            net_input = x[:, :, i - 3:i + 4]
                            net_input = torch.from_numpy(np.ascontiguousarray(net_input)).permute(2, 0,1).float().unsqueeze(0)
                            net_input = net_input.to(device)
                            Nsigma = torch.full((1, 1, 1, 1), h_ch / 255.).type_as(net_input)
                            output = model(net_input, Nsigma)
                            output = output.data.squeeze().cpu().numpy()
                            tem = np.dstack((tem, output))
                            nsig = ori_nsig
                    #x = np.clip(tem,0,1)
                    x=tem

                else:
                    x = denoise_tv_chambolle(x, nsig / 255, max_num_iter=tv_iter_max, channel_axis=-1 if multichannel else None)
                    #x = TV_denoiser(x, tv_weight, n_iter_max=tv_iter_max)
            elif denoiser.lower() =='bm3d':
                sigma = nsig/255
                v = np.zeros((15, 15))
                for x1 in range(-7, 8, 1):
                    for x2 in range(-7, 8, 1):
                        v[x1 + 7, x2 + 7] = 1 / (x1 ** 2 + x2 ** 2 + 1)
                v = v / np.sum(v)
                for i in range(8):
                    x[:,:,i]= bm3d_deblurring(np.atleast_3d(x[:,:,i]), sigma, v)
            else:
                raise ValueError('Unsupported denoiser {}!'.format(denoiser))
            # [optional] calculate image quality assessment, i.e., PSNR for 
            # every five iterations
            if show_iqa and X_orig is not None:
                ssim_all.append(calculate_ssim(X_orig, x))
                psnr_all.append(psnr(X_orig, x))
                if (k+1)%1 == 0:
                    if not noise_estimate and nsig is not None:
                        if nsig < 1:
                            print('  GAP-{0} iteration {1: 3d}, sigma {2: 3g}/255, ' 
                              'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                               k+1, nsig*255, psnr_all[k]),
                              'SSIM:{}'.format(ssim_all[k]))
                        else:
                            print('  GAP-{0} iteration {1: 3d}, sigma {2: 3g}, ' 
                                'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                                k+1, nsig, psnr_all[k]),
                              'SSIM:{}'.format(ssim_all[k]))
                    else:
                        print('  GAP-{0} iteration {1: 3d}, ' 
                              'PSNR {2:2.2f} dB.'.format(denoiser.upper(), 
                               k+1, psnr_all[k]),
                              'SSIM:{}'.format(ssim_all[k]))
            if k==1000:
                break
            k = k+1

    return x, psnr_all

def GAP_TV_rec(y,Phi,A, At,Phi_sum, maxiter, step_size, weight, row, col, ColT, X_ori):
    y1 = np.zeros((row,col))
    begin_time = time.time()
    f = At(y,Phi)
    for ni in range(maxiter):
        fb = A(f,Phi)
        y1 = y1+ (y-fb)
        f  = f + np.multiply(step_size, At( np.divide(y1-fb,Phi_sum),Phi ))
        f = denoise_tv_chambolle(f, weight,max_num_iter=30,channel_axis=True)
    
        if (ni+1)%5 == 0:
            # mse = np.mean(np.sum((y-A(f,Phi))**2,axis=(0,1)))
            end_time = time.time()
            print("GAP-TV: Iteration %3d, PSNR = %2.2f dB,"
              " time = %3.1fs."
              % (ni+1, psnr(f, X_ori), end_time-begin_time))
    return f
