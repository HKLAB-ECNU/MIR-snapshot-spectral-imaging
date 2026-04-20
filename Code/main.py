# We thank Prof. Xin Yuan from Westlake University for sharing the reconstruction code on GitHub (https://github.com/zsm1211/PnP-CASSI).
# Our implementation is adapted and modified from his original codebase, and we have made use of the GAP-TV algorithm.
import os
import time
import math
import h5py
import numpy as np
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
from statistics import mean
from numpy import *
from PIL import Image
import scipy.ndimage as ndimage
from scipy.ndimage import zoom
from dvp_linear_inv_cassi import (gap_denoise)
from utils import (A, At, psnr,shift,shift_back)
from scipy.stats import pearsonr

#Imaging wavelength, dispersion scaling factor, dispersion center definition
MIR_lambda = []
for i in range(8):
    MIR_lambda.append(2500+65*(i))
MIR_lambda = np.array(MIR_lambda)
Pump_lambda = 1030
f2 = 75
f1 = 50
M = f2/f1*Pump_lambda/(MIR_lambda+Pump_lambda)
r, c, nC = 512, 512, len(MIR_lambda)
x0 = 255.5
y0 = 255.5
print(M)

#Imaging object
Object = Image.open(r'.\Object\Object.png')
Object = Object.convert('L')
Object = np.array(Object)
Object = Object[11:523,25:537]
Object = Object*(-1)+255
Object = cv2.resize(Object, (r, c), interpolation=cv2.INTER_LINEAR)
gaussian = np.zeros((r,c))
for ii in range(r):
    for jj in range(c):
        gaussian[ii,jj] = np.exp(-2*(np.square(ii-x0)+np.square(jj-y0))/x0/y0/2/2)
        Object[ii, jj] = Object[ii, jj] * gaussian[ii,jj]
Object_3D_a = np.zeros((r,c,nC))
Object_3D= np.zeros((r,c,nC))
plt.figure()
plt.imshow(Object,cmap='gray')
plt.tight_layout()
plt.axis('off')
plt.show()

#Encoding matrix
mask=sio.loadmat(r'.\Mask\mask512.mat')['mask']
mask = np.array(mask)
plt.figure()
plt.imshow(mask,cmap=plt.cm.gray)
plt.show()
Mask_3D= np.zeros((r,c,nC))

#Image dispersion
for n in range (nC):
    Object_dispersion = np.zeros((r, c))
    for ii in range(r):
        for jj in range(c):
            h  = int(M[n]/M[0]*(ii-x0)+x0)
            w =  int(M[n]/M[0]*(jj-y0)+y0)
            Object_dispersion[h,w] = Object[ii,jj]
    Object_3D[:,:,n] = Object_dispersion*(-np.square((n-4))*0.035+1)

#Mask dispersion
for n in range (nC):
    Mask_dispersion = np.zeros((r, c))
    for ii in range(r):
        for jj in range(c):
            h = int(M[n]/M[0]* (ii - x0) + x0 - 0.5)
            w = int(M[n]/M[0]* (jj - y0) + y0 - 0.5)
            Mask_dispersion[h, w] = mask[ii, jj]
    Mask_3D[:,:,n] = Mask_dispersion

#Measurement image
meas = np.sum(Mask_3D*Object_3D,2)
plt.figure()
plt.imshow(meas,cmap='gray')
plt.tight_layout()
plt.axis('off')
plt.show()

#Image decoupling
method = 'GAP'
Phi = Mask_3D
Phi_sum = np.sum(Mask_3D**2,2)
Phi_sum[Phi_sum==0]=1

if method == 'GAP':
    _lambda = 1 # regularization factor
    accelerate = True # enable accelerated version of GAP
    denoiser = 'tv' # total variation (TV); deep denoiser(hsicnn)
    iter_max = 1000 # maximum number of iterations
    tv_weight = 6 # TV denoising weight (larger for smoother but slower)
    tv_iter_max = 5 # TV denoising maximum number of iterations each
    begin_time = time.time()
    vgaptv,psnr_gaptv = gap_denoise(meas,Phi,A,At,_lambda,
                        accelerate, denoiser, iter_max,
                        tv_weight=tv_weight,
                        tv_iter_max=tv_iter_max,
                        X_orig=Object_3D,sigma=[130,130])#
    end_time = time.time()
    tgaptv = end_time - begin_time
    print(psnr_gaptv)
    print(tgaptv)
for i in range(nC):
    wavelength = 2600 + i * 65
    plt.subplot(2, nC, i + 1)
    plt.imshow(vgaptv[:, :, i], cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title(f'Rec. {wavelength} nm')
    plt.axis('off')

    plt.subplot(2, nC, i + nC + 1)
    plt.imshow(Object_3D[:, :, i], cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title(f' Ori. {wavelength} nm')
    plt.axis('off')
plt.show()
