#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For photometry data pre-processing
Modified from pyPhotometry pre-processing (by Akam)

For 4-Fiber brances
Column# in CSV Data
Loc0:3 -> 8
Loc1:4 -> 9
Loc2:5 -> 10
Loc3:6 -> 11
______________
22061x-
8,9,10,11

Started on Fri Mar  4 01:58:25 2022
@author: Kenta M. Hagihara @SvobodaLab
"""
# clear all
from IPython import get_ipython
get_ipython().magic("reset -sf")

#%% 
import os
import csv
import numpy as  np
import pylab as plt
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize
import glob

#%% import 
plt.close('all')

#Win
AnalDir = r"F:\photometry_FIPopt\230428\669485"


nFibers = 2
nColor = 3
sampling_rate = 20 #individual channel (not total)
nFrame2cut = 100  #crop initial n frames
b_percentile = 0.70 #To calculare F0, median of bottom x%

  
#BiExpFitIni = [1,1e-3,5,1e-3,5]
BiExpFitIni = [1,1e-3,1,1e-3,1]  #currentlu not used

if bool(glob.glob(AnalDir + os.sep + "L470*")) == True:
    print('preprocessing Neurophotometrics Data')
    file1  = glob.glob(AnalDir + os.sep + "L415*")[0]
    file2 = glob.glob(AnalDir + os.sep + "L470*")[0]
    file3 = glob.glob(AnalDir + os.sep + "L560*")[0]
    
elif bool(glob.glob(AnalDir + os.sep + "FIP_DataG*")) == True:
    print('preprocessing FIP Data')
    file1  = glob.glob(AnalDir + os.sep + "FIP_DataIso*")[0]
    file2 = glob.glob(AnalDir + os.sep + "FIP_DataG*")[0]
    file3 = glob.glob(AnalDir + os.sep + "FIP_DataR*")[0]
    
else:
    print('photometry raw data missing; please check the folder specified as AnalDir')
          
with open(file1) as f:
    reader = csv.reader(f)
    datatemp = np.array([row for row in reader])
    data1 = datatemp[1:,:].astype(np.float32)
    #del datatemp
    
with open(file2) as f:
    reader = csv.reader(f)
    datatemp = np.array([row for row in reader])
    data2 = datatemp[1:,:].astype(np.float32)
    #del datatemp
    
with open(file3) as f:
    reader = csv.reader(f)
    datatemp = np.array([row for row in reader])
    data3 = datatemp[1:,:].astype(np.float32)
    #del datatemp
        
# in case acquisition halted accidentally
Length = np.amin([len(data1),len(data2),len(data3)])
data1 = data1[0:Length] 
data2 = data2[0:Length]
data3 = data3[0:Length]
 
#%% Data sort # 1,2:Time-Frame info; ROI0:3;ROI1:4,ROI2:5,ROI3:6;...
#Data_Fiber1iso = data1[:,3]
#Data_Fiber1G = data2[:,3]
#Data_Fiber1R = data3[:,5]
 
#Data_Fiber2iso = data1[:,4]
#Data_Fiber2G = data2[:,4]
#Data_Fiber2R = data3[:,6]
#%% from 220609-, ROI0:8;ROI1:9,ROI2:10,ROI3:11;

if bool(glob.glob(AnalDir + os.sep + "L470*")) == True:
    Data_Fiber1iso = data1[:,8]
    Data_Fiber1G = data2[:,8]
    Data_Fiber1R = data3[:,10]
     
    Data_Fiber2iso = data1[:,9]
    Data_Fiber2G = data2[:,9]
    Data_Fiber2R = data3[:,11]

elif bool(glob.glob(AnalDir + os.sep + "FIP_DataG*")) == True:
    PMts= data2[:,0]
    Data_Fiber1iso = data1[:,1]
    Data_Fiber1G = data2[:,1]
    Data_Fiber1R = data3[:,1]
     
    Data_Fiber2iso = data1[:,2]
    Data_Fiber2G = data2[:,2]
    Data_Fiber2R = data3[:,2]
    
time_seconds = np.arange(len(Data_Fiber1iso)) /sampling_rate 
    
# Visualizing raw data (still including excitation-off )
plt.figure()
plt.subplot(2,1,1)
plt.plot(time_seconds, Data_Fiber1G, 'g', label='G1')
plt.plot(time_seconds, Data_Fiber1R, 'r', label='R1')
plt.plot(time_seconds, Data_Fiber1iso, 'b', label='iso1')
plt.xlabel('Time (seconds)')
plt.ylabel('CMOS Signal')
plt.title('Raw signals:ROI1')
plt.legend()
plt.subplot(2,1,2)
plt.plot(time_seconds, Data_Fiber2G, 'g', label='G2')
plt.plot(time_seconds, Data_Fiber2R, 'r', label='R2')
plt.plot(time_seconds, Data_Fiber2iso, 'b', label='iso2')
plt.xlabel('Time (seconds)')
plt.ylabel('CMOS Signal')
plt.title('Raw signals:ROI2')
plt.legend()

#%% From here to be multiplexed

#cropping
G1_raw = Data_Fiber1G[nFrame2cut:]
G2_raw = Data_Fiber2G[nFrame2cut:]
R1_raw = Data_Fiber1R[nFrame2cut:]
R2_raw = Data_Fiber2R[nFrame2cut:]
Ctrl1_raw = Data_Fiber1iso[nFrame2cut:]
Ctrl2_raw = Data_Fiber2iso[nFrame2cut:]

time_seconds = np.arange(len(G1_raw)) /sampling_rate 

#%% Raw signals
plt.figure()
plt.subplot(2,1,1)
plt.plot(time_seconds, G1_raw, 'g', label='G1')
plt.plot(time_seconds, R1_raw, 'r', label='R1')
plt.plot(time_seconds, Ctrl1_raw, 'b', label='iso')
plt.xlabel('Time (seconds)')
plt.ylabel('CMOS Signal')
plt.title('Raw signals:ROI1')
plt.legend()

plt.subplot(2,1,2)
plt.plot(time_seconds, G2_raw, 'g', label='G2')
plt.plot(time_seconds, R2_raw, 'r', label='R2')
plt.plot(time_seconds, Ctrl2_raw, 'b', label='R2')
plt.xlabel('Time (seconds)')
plt.ylabel('CMOS Signal')
plt.title('Raw signals:ROI12')
plt.tight_layout()
plt.legend()

#%% Median filtering to remove electrical artifact.
kernelSize=1
G1_denoised = medfilt(G1_raw, kernel_size=kernelSize)
G2_denoised = medfilt(G2_raw, kernel_size=kernelSize)
R1_denoised = medfilt(R1_raw, kernel_size=kernelSize)
R2_denoised = medfilt(R2_raw, kernel_size=kernelSize)
Ctrl1_denoised = medfilt(Ctrl1_raw, kernel_size=kernelSize)
Ctrl2_denoised = medfilt(Ctrl2_raw, kernel_size=kernelSize)
 
# Lowpass filter - zero phase filtering (with filtfilt) is used to avoid distorting the signal.
b,a = butter(2, 9, btype='low', fs=sampling_rate)
G1_denoised = filtfilt(b,a, G1_denoised)
G2_denoised = filtfilt(b,a, G2_denoised)
R1_denoised = filtfilt(b,a, R1_denoised)
R2_denoised = filtfilt(b,a, R2_denoised)
Ctrl1_denoised = filtfilt(b,a, Ctrl1_denoised)
Ctrl2_denoised = filtfilt(b,a, Ctrl2_denoised)

plt.figure()
plt.subplot(2,1,1)
plt.plot(time_seconds, G1_denoised, 'g', label='G1 denoised')
plt.plot(time_seconds, R1_denoised, 'r', label='R1 denoised')
plt.plot(time_seconds, Ctrl1_denoised, 'b', label='iso1 denoised') 
plt.title('Denoised signals:ROI1')
plt.tight_layout()
plt.legend()

plt.subplot(2,1,2)
plt.plot(time_seconds, G2_denoised, 'g', label='G1 denoised')
plt.plot(time_seconds, R2_denoised, 'r', label='R1 denoised')
plt.plot(time_seconds, Ctrl2_denoised, 'b', label='iso1 denoised') 
plt.title('Denoised signals:ROI2')
plt.tight_layout()
plt.legend()

#%% Photobleaching correction by LowCut
'''
b,a = butter(2, 0.05, btype='high', fs=sampling_rate)
G1_highpass = filtfilt(b,a, G1_denoised, padtype='even')
G2_highpass = filtfilt(b,a, G2_denoised, padtype='even')
R1_highpass = filtfilt(b,a, R1_denoised, padtype='even')
R2_highpass = filtfilt(b,a, R2_denoised, padtype='even')
Ctrl1_highpass = filtfilt(b,a, Ctrl1_denoised, padtype='even')
Ctrl2_highpass = filtfilt(b,a, Ctrl2_denoised, padtype='even')

plt.figure()
plt.subplot(1,2,1)
plt.plot(time_seconds, G1_highpass,'g', label='G1 highpass')
plt.plot(time_seconds, R1_highpass,'r', label='R1 highpass')
plt.plot(time_seconds, Ctrl1_highpass,'b', label='iso1 highpass')

plt.subplot(1,2,2)
plt.plot(time_seconds, G2_highpass,'g', label='G2 highpass')
plt.plot(time_seconds, R2_highpass,'r', label='R2 highpass')
plt.plot(time_seconds, Ctrl2_highpass,'b', label='iso2 highpass')
plt.xlabel('Time (seconds)')
plt.ylabel('CMOS Signal')
plt.title('Bleaching correction by highpass filtering')
plt.legend();
'''
#%% Bi-exponential curve fit.
'''
#def exp_func(x, a, b, c):
#   return a*np.exp(-b*x) + c
BiExpFitIni = [1,1e-3,5,1e-3,5]


def biexpF(x, a, b, c, d, e):
    return a * np.exp(-b * x) + c * np.exp(-d * x) + e

# Fit curve to signals.
G1_parms, parm_cov1 = curve_fit(biexpF, time_seconds, G1_denoised, p0=BiExpFitIni,maxfev=5000)
G1_expfit = biexpF(time_seconds, *G1_parms)
G2_parms, parm_cov2 = curve_fit(biexpF, time_seconds, G2_denoised, p0=BiExpFitIni,maxfev=5000)
G2_expfit = biexpF(time_seconds, *G2_parms)
#R1_parms, parm_cov1 = curve_fit(biexpF, time_seconds, R1_denoised, p0=BiExpFitIni,maxfev=5000)
#R1_expfit = biexpF(time_seconds, *R1_parms)
R2_parms, parm_cov2 = curve_fit(biexpF, time_seconds, R2_denoised, p0=BiExpFitIni,maxfev=5000)
R2_expfit = biexpF(time_seconds, *R2_parms)

# Fit curve to ctrl.
Ctrl1_parms, parm_cov = curve_fit(biexpF, time_seconds, Ctrl1_denoised, p0=BiExpFitIni,maxfev=5000)
Ctrl1_expfit = biexpF(time_seconds, *Ctrl1_parms)
Ctrl2_parms, parm_cov = curve_fit(biexpF, time_seconds, Ctrl2_denoised, p0=BiExpFitIni,maxfev=5000)
Ctrl2_expfit = biexpF(time_seconds, *Ctrl2_parms)

plt.figure()
plt.subplot(1,2,1)
plt.plot(time_seconds, G1_denoised, 'g', label='G1_denoised')
plt.plot(time_seconds, R1_denoised, 'r', label='R1_denoised')
plt.plot(time_seconds, Ctrl1_denoised, 'b', label='iso1_denoised')
plt.plot(time_seconds, G1_expfit,'k', linewidth=1.5) 
#plt.plot(time_seconds, R1_expfit,'k', linewidth=1.5) 
plt.plot(time_seconds, Ctrl1_expfit,'k', linewidth=1.5) 
plt.title('Bi-exponential fit to bleaching.')
plt.xlabel('Time (seconds)');

plt.subplot(1,2,2)
plt.plot(time_seconds, G2_denoised, 'g', label='G2_denoised')
plt.plot(time_seconds, R2_denoised, 'r', label='R2_denoised')
plt.plot(time_seconds, Ctrl2_denoised, 'b', label='iso2_denoised')
plt.plot(time_seconds, G2_expfit,'k', linewidth=1.5) 
plt.plot(time_seconds, R2_expfit,'k', linewidth=1.5) 
plt.plot(time_seconds, Ctrl2_expfit,'k', linewidth=1.5) 
plt.title('Bi-exponential fit to bleaching.')
plt.xlabel('Time (seconds)');

G1_es = G1_denoised - G1_expfit
G2_es = G2_denoised - G2_expfit
#R1_es = R1_denoised - R1_expfit
R2_es = R2_denoised - R2_expfit
Ctrl1_es = Ctrl1_denoised - Ctrl1_expfit
Ctrl2_es = Ctrl2_denoised - Ctrl2_expfit

plt.figure()
plt.subplot(1,2,1)
plt.plot(time_seconds, G1_es, 'g', label='G1')
#plt.plot(time_seconds, R1_es, 'r', label='R1')
plt.plot(time_seconds, Ctrl1_es, 'b', label='iso1')
plt.title('Bleaching correction by subtraction of biexponential fit')
plt.xlabel('Time (seconds)');
plt.subplot(1,2,2)
plt.plot(time_seconds, G2_es, 'g', label='G2')
plt.plot(time_seconds, R2_es, 'r', label='R2')
plt.plot(time_seconds, Ctrl2_es, 'b', label='iso2')
plt.title('Bleaching correction by subtraction of biexponential fit')
plt.xlabel('Time (seconds)');

'''
#%%
# Fit 4th order polynomial to signals.
coefs_G1 = np.polyfit(time_seconds, G1_denoised, deg=4)
G1_polyfit = np.polyval(coefs_G1, time_seconds)
coefs_G2 = np.polyfit(time_seconds, G2_denoised, deg=4)
G2_polyfit = np.polyval(coefs_G2, time_seconds)
coefs_R1 = np.polyfit(time_seconds, R1_denoised, deg=4)
R1_polyfit = np.polyval(coefs_R1, time_seconds)
coefs_R2 = np.polyfit(time_seconds, R2_denoised, deg=4)
R2_polyfit = np.polyval(coefs_R2, time_seconds)

# Fit 4th order polynomial to Ctrl.
coefs_Ctrl1 = np.polyfit(time_seconds, Ctrl1_denoised, deg=4)
Ctrl1_polyfit = np.polyval(coefs_Ctrl1, time_seconds)
coefs_Ctrl2 = np.polyfit(time_seconds, Ctrl2_denoised, deg=4)
Ctrl2_polyfit = np.polyval(coefs_Ctrl2, time_seconds)

# Plot fits
plt.figure()
plt.subplot(2,1,1)
plt.plot(time_seconds, G1_denoised, 'g', label='G1_denoised')
plt.plot(time_seconds, R1_denoised, 'r', label='R1_denoised')
plt.plot(time_seconds, Ctrl1_denoised, 'b', label='iso1_denoised')
plt.plot(time_seconds, G1_polyfit,'k', linewidth=1.5) 
plt.plot(time_seconds, R1_polyfit,'k', linewidth=1.5) 
plt.plot(time_seconds, Ctrl1_polyfit,'k', linewidth=1.5) 
plt.title('polyfi ROI1')
plt.xlabel('Time (seconds)');

plt.subplot(2,1,2)
plt.plot(time_seconds, G2_denoised, 'g', label='G2_denoised')
plt.plot(time_seconds, R2_denoised, 'r', label='R2_denoised')
plt.plot(time_seconds, Ctrl2_denoised, 'b', label='iso2_denoised')
plt.plot(time_seconds, G2_polyfit,'k', linewidth=1.5) 
plt.plot(time_seconds, R2_polyfit,'k', linewidth=1.5) 
plt.plot(time_seconds, Ctrl2_polyfit,'k', linewidth=1.5) 
plt.title('polyfit ROI2')
plt.xlabel('Time (seconds)');


# "_es for fitted curved subtracted signals"
G1_es = G1_denoised - G1_polyfit
G2_es = G2_denoised - G2_polyfit
R1_es = R1_denoised - R1_polyfit
R2_es = R2_denoised - R2_polyfit
Ctrl1_es = Ctrl1_denoised - Ctrl1_polyfit
Ctrl2_es = Ctrl2_denoised - Ctrl2_polyfit

plt.figure()
plt.subplot(2,1,1)
plt.plot(time_seconds, Ctrl1_es, 'b', label='iso1_estim')
plt.plot(time_seconds, G1_es, 'g', label='G1_estim')
plt.plot(time_seconds, R1_es, 'r', label='R1_estim')
plt.title('polyfit ROI1')

plt.subplot(2,1,2)
plt.plot(time_seconds, Ctrl2_es, 'b', label='iso2_estim')
plt.plot(time_seconds, G2_es, 'g', label='G2_estim')
plt.plot(time_seconds, R2_es, 'r', label='R2_estim')

plt.title('polyfit ROI2')
plt.xlabel('Time (seconds)');

#%%Additional LowCut
'''
b,a = butter(2, 0.1, btype='high', fs=sampling_rate)
G1_es2 = filtfilt(b,a, G1_es, padtype='even')
G2_es2 = filtfilt(b,a, G2_es, padtype='even')
R1_es2 = filtfilt(b,a, R1_es, padtype='even')
R2_es2 = filtfilt(b,a, R2_es, padtype='even')
Ctrl1_es2 = filtfilt(b,a, Ctrl1_es, padtype='even')
Ctrl2_es2 = filtfilt(b,a, Ctrl2_es, padtype='even')

plt.figure()
plt.subplot(1,2,1)
plt.plot(time_seconds, G1_es2,'g', label='G1 highpass')
plt.plot(time_seconds, R1_es2,'r', label='R1 highpass')
plt.plot(time_seconds, Ctrl1_es2,'b', label='iso1 highpass')

plt.subplot(1,2,2)
plt.plot(time_seconds, G2_es2,'g', label='G2 highpass')
plt.plot(time_seconds, R2_es2,'r', label='R2 highpass')
plt.plot(time_seconds, Ctrl2_es2,'b', label='iso2 highpass')
plt.xlabel('Time (seconds)')
plt.ylabel('CMOS Signal')
plt.title('Bleaching correction by highpass filtering')
plt.legend();
'''

#%% Motion correction using iso
slopeG1, interceptG1, r_valueG1, p_valueG1, std_errG1 = linregress(x=Ctrl1_es, y=G1_es)
slopeG2, interceptG2, r_valueG2, p_valueG2, std_errG2 = linregress(x=Ctrl2_es, y=G2_es)
slopeR1, interceptR1, r_valueR1, p_valueR1, std_errR1 = linregress(x=Ctrl1_es, y=R1_es)
slopeR2, interceptR2, r_valueR2, p_valueR2, std_errR2 = linregress(x=Ctrl2_es, y=R2_es)

plt.figure()
plt.subplot(2,2,1)
plt.scatter(Ctrl1_es[::5], G1_es[::5],alpha=0.1, marker='.')
x = np.array(plt.xlim())
plt.plot(x, interceptG1+slopeG1*x)
plt.xlabel('iso1')
plt.ylabel('G1')
plt.title('iso - G correlation.')

plt.subplot(2,2,2)
plt.scatter(Ctrl1_es[::5], G1_es[::5],alpha=0.1, marker='.')
x = np.array(plt.xlim())
plt.plot(x, interceptG1+slopeG1*x)
plt.xlabel('iso2')
plt.ylabel('G2')
plt.title('iso - G correlation.')

plt.subplot(2,2,3)
plt.scatter(Ctrl1_es[::5], R1_es[::5],alpha=0.1, marker='.')
x = np.array(plt.xlim())
plt.plot(x, interceptR1+slopeR1*x)
plt.xlabel('iso1')
plt.ylabel('R1')
plt.title('iso - R correlation.')

plt.subplot(2,2,4)
plt.scatter(Ctrl2_es[::5], R2_es[::5],alpha=0.1, marker='.')
x = np.array(plt.xlim())
plt.plot(x, interceptR2+slopeR2*x)
plt.xlabel('iso2')
plt.ylabel('R2')
plt.title('iso - R correlation.')
plt.tight_layout()

print('SlopeG1    : {:.3f}'.format(slopeG1))
print('R-squaredG1: {:.3f}'.format(r_valueG1**2))
print('SlopeG2    : {:.3f}'.format(slopeG2))
print('R-squaredG2: {:.3f}'.format(r_valueG2**2))

print('SlopeR1    : {:.3f}'.format(slopeR1))
print('R-squaredR1: {:.3f}'.format(r_valueR1**2))
print('SlopeR2    : {:.3f}'.format(slopeR2))
print('R-squaredR2: {:.3f}'.format(r_valueR2**2))

#% motion corrected ("corrected" currently not used to be conservative)
G1_est_motion = interceptG1 + slopeG1 * Ctrl1_es
G1_corrected = G1_es - G1_est_motion
G2_est_motion = interceptG2 + slopeG2 * Ctrl2_es
G2_corrected = G2_es - G2_est_motion

R1_est_motion = interceptR1 + slopeR1 * Ctrl1_es
R1_corrected = R1_es - G1_est_motion
R2_est_motion = interceptR2 + slopeR2 * Ctrl2_es
R2_corrected = R2_es - R2_est_motion

plt.figure()
plt.subplot(2,2,1)
plt.plot(time_seconds, G1_es , label='G1 - pre motion correction')
plt.plot(time_seconds, G1_corrected, 'g', label='G1 - motion corrected')
plt.plot(time_seconds, G1_est_motion, 'y', label='estimated motion')
plt.xlabel('Time (seconds)')
plt.title('Motion correction G1')
plt.legend()

plt.subplot(2,2,2)
plt.plot(time_seconds, G2_es , label='G2 - pre motion correction')
plt.plot(time_seconds, G2_corrected, 'g', label='G2 - motion corrected')
plt.plot(time_seconds, G2_est_motion, 'y', label='estimated motion')
plt.xlabel('Time (seconds)')
plt.title('Motion correction G2')
plt.legend()

plt.subplot(2,2,3)
plt.plot(time_seconds, R1_es , label='R1 - pre motion correction')
plt.plot(time_seconds, R1_corrected, 'r', label='R1 - motion corrected')
plt.plot(time_seconds, R1_est_motion, 'y', label='estimated motion')
plt.xlabel('Time (seconds)')
plt.title('Motion correction R1')
plt.legend()

plt.subplot(2,2,4)
plt.plot(time_seconds, R2_es , label='R2 - pre motion correction')
plt.plot(time_seconds, R2_corrected, 'r', label='R2 - motion corrected')
plt.plot(time_seconds, R2_est_motion, 'y', label='estimated motion')
plt.xlabel('Time (seconds)')
plt.title('Motion correction R2')
plt.legend()

plt.tight_layout()

#%% Calculating sliding baseline for dFF
b,a = butter(2, 0.0001, btype='low', fs=sampling_rate)
G1_baseline = filtfilt(b,a, G1_denoised, padtype='even')
G2_baseline = filtfilt(b,a, G2_denoised, padtype='even')
R1_baseline = filtfilt(b,a, R1_denoised, padtype='even')
R2_baseline = filtfilt(b,a, R2_denoised, padtype='even')

Ctrl1_baseline = filtfilt(b,a, Ctrl1_denoised, padtype='even')
Ctrl2_baseline = filtfilt(b,a, Ctrl2_denoised, padtype='even')

plt.figure()
plt.subplot(2,1,1)
plt.plot(time_seconds, G1_baseline, 'g', label='baselineG1')
plt.plot(time_seconds, R1_baseline, 'r', label='baselineR1')
plt.plot(time_seconds, Ctrl1_baseline,'b', label='baselineCtrl1')
plt.xlabel('Time (seconds)')
plt.title('sliding baseline')
plt.legend()

plt.subplot(2,1,2)
plt.plot(time_seconds, G2_baseline, 'g', label='baselineG2')
plt.plot(time_seconds, R2_baseline, 'r', label='baselineR2')
plt.plot(time_seconds, Ctrl2_baseline,'b', label='baselineCtrl2')
plt.xlabel('Time (seconds)')
plt.title('sliding baseline')
plt.legend()
plt.tight_layout()

#%% dF/F calculation (note using _es insted of corrected)
#G1_dF_F = G1_corrected/G1_baseline
G1_dF_F = G1_es/G1_baseline
sort = np.sort(G1_dF_F)
b_median = np.median(sort[0:round(len(sort) * b_percentile)])
G1_dF_F = G1_dF_F - b_median

#G2_dF_F = G2_corrected/G2_baseline
G2_dF_F = G2_es/G2_baseline
sort = np.sort(G2_dF_F)
b_median = np.median(sort[0:round(len(sort) * b_percentile)])
G2_dF_F = G2_dF_F - b_median

#R1_dF_F = R1_corrected/R1_baseline
R1_dF_F = R1_es/R1_baseline
sort = np.sort(R1_dF_F)
b_median = np.median(sort[0:round(len(sort) * b_percentile)])
R1_dF_F = R1_dF_F - b_median

#R2_dF_F = R2_corrected/R2_baseline
R2_dF_F = R2_es/R2_baseline
sort = np.sort(R2_dF_F)
b_median = np.median(sort[0:round(len(sort) * b_percentile)])
R2_dF_F = R2_dF_F - b_median

Ctrl1_dF_F = Ctrl1_es/Ctrl1_baseline
sort = np.sort(Ctrl1_dF_F)
b_median = np.median(sort[0:round(len(sort) * b_percentile)])
Ctrl1_dF_F = Ctrl1_dF_F - b_median

Ctrl2_dF_F = Ctrl2_es/Ctrl2_baseline
sort = np.sort(Ctrl2_dF_F)
b_median = np.median(sort[0:round(len(sort) * b_percentile)])
Ctrl2_dF_F = Ctrl2_dF_F - b_median


#%% fixing the cropped length by filling 
G1_dF_F = np.append(np.ones([nFrame2cut,1])*G1_dF_F[0],G1_dF_F)
G2_dF_F = np.append(np.ones([nFrame2cut,1])*G2_dF_F[0],G2_dF_F)
R1_dF_F = np.append(np.ones([nFrame2cut,1])*R1_dF_F[0],R1_dF_F)
R2_dF_F = np.append(np.ones([nFrame2cut,1])*R2_dF_F[0],R2_dF_F)

Ctrl1_dF_F = np.append(np.ones([nFrame2cut,1])*Ctrl1_dF_F[0],Ctrl1_dF_F)
Ctrl2_dF_F = np.append(np.ones([nFrame2cut,1])*Ctrl2_dF_F[0],Ctrl2_dF_F)

#%%
time_seconds = np.arange(len(G1_dF_F)) /sampling_rate 

plt.figure()
plt.subplot(3,2,1)
plt.plot(time_seconds, G1_dF_F*100, 'g')
plt.plot(time_seconds, np.zeros(len(time_seconds)),'--k')
plt.xlabel('Time (seconds)')
plt.ylabel('G dF/F (%)')
plt.title('G1 dF/F')
plt.grid(True)

plt.subplot(3,2,2)
plt.plot(time_seconds, G2_dF_F*100, 'g')
plt.plot(time_seconds, np.zeros(len(time_seconds)),'--k')
plt.xlabel('Time (seconds)')
plt.ylabel('G dF/F (%)')
plt.title('G2 dF/F')
plt.grid(True)

plt.subplot(3,2,3)
plt.plot(time_seconds, R1_dF_F*100, 'r')
plt.plot(time_seconds, np.zeros(len(time_seconds)),'--k')
plt.xlabel('Time (seconds)')
plt.ylabel('R dF/F (%)')
plt.title('R1 dF/F')
plt.grid(True)

plt.subplot(3,2,4)
plt.plot(time_seconds, R2_dF_F*100, 'r')
plt.plot(time_seconds, np.zeros(len(time_seconds)),'--k')
plt.xlabel('Time (seconds)')
plt.ylabel('R dF/F (%)')
plt.title('R2 dF/F')
plt.grid(True)

plt.subplot(3,2,5)
plt.plot(time_seconds, Ctrl1_dF_F*100, 'b')
plt.plot(time_seconds, np.zeros(len(time_seconds)),'--k')
plt.xlabel('Time (seconds)')
plt.ylabel('Ctrl dF/F (%)')
plt.title('Ctrl1 dF/F')
plt.grid(True)

plt.subplot(3,2,6)
plt.plot(time_seconds, Ctrl2_dF_F*100, 'b')
plt.plot(time_seconds, np.zeros(len(time_seconds)),'--k')
plt.xlabel('Time (seconds)')
plt.ylabel('Ctrl dF/F (%)')
plt.title('Ctrl2 dF/F')
plt.grid(True)
plt.tight_layout()




#%% Save
np.save(AnalDir + os.sep + "G1_dF_F", G1_dF_F)
np.save(AnalDir + os.sep + "G2_dF_F", G2_dF_F)
np.save(AnalDir + os.sep + "R1_dF_F", R1_dF_F)
np.save(AnalDir + os.sep + "R2_dF_F", R2_dF_F)
np.save(AnalDir + os.sep + "Ctrl1_dF_F", Ctrl1_dF_F)
np.save(AnalDir + os.sep + "Ctrl2_dF_F", Ctrl2_dF_F)

if bool(glob.glob(AnalDir + os.sep + "FIP_DataG*")) == True:
    np.save(AnalDir + os.sep + "PMts", PMts)


#%%
plt.figure()
plt.plot(time_seconds, R1_dF_F*100, 'magenta')
plt.plot(time_seconds, G1_dF_F*100, 'g')
plt.xlabel('Time (seconds)')
plt.ylabel('dF/F (%)')
plt.title('L:')

plt.figure()
plt.plot(time_seconds, R2_dF_F*100, 'magenta')
plt.plot(time_seconds, G2_dF_F*100, 'g')
plt.xlabel('Time (seconds)')
plt.ylabel('dF/F (%)')
plt.title('R:')


#%%
kernelSize=51

plt.figure()
plt.subplot(5,1,1)
G1norm = G1_dF_F / np.max(G1_dF_F)
G1norm = medfilt(G1norm, kernel_size=kernelSize)
G2norm = G2_dF_F / np.max(G2_dF_F)
G2norm = medfilt(G2norm, kernel_size=kernelSize)
R1norm = R1_dF_F / np.max(R1_dF_F)
R1norm = medfilt(R1norm, kernel_size=kernelSize)
R2norm = R2_dF_F / np.max(R2_dF_F)
R2norm = medfilt(R2norm, kernel_size=kernelSize)

plt.subplot(5,1,1)
plt.plot(time_seconds, G1norm / np.max(G1norm), 'green')
plt.plot(time_seconds, R1norm / np.max(R1norm), 'magenta')

plt.plot(time_seconds, G2norm / np.max(G2norm), 'blue')
plt.plot(time_seconds, R2norm / np.max(R2norm), 'red')

plt.subplot(5,1,2)
plt.plot(time_seconds, G1norm / np.max(G1norm), 'green')
plt.subplot(5,1,3)
plt.plot(time_seconds, R1norm / np.max(R1norm), 'magenta')
plt.subplot(5,1,4)
plt.plot(time_seconds, G2norm / np.max(G2norm), 'blue')
plt.subplot(5,1,5)
plt.plot(time_seconds, R2norm / np.max(R2norm), 'red')


#%%
plt.figure()
plt.subplot(2,1,1)
plt.plot(time_seconds, R1_dF_F*100, 'blue')
plt.plot(time_seconds, R2_dF_F*100, 'red')

plt.subplot(2,1,2)
plt.plot(time_seconds, G1_dF_F*100, 'blue')
plt.plot(time_seconds, G2_dF_F*100, 'red')

np.corrcoef(G1_dF_F, G2_dF_F)
np.corrcoef(R1_dF_F, R2_dF_F)




