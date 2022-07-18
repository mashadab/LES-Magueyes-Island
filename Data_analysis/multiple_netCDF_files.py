#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:06:39 2022

@author: afzal-admin
"""

#Login to compute node
# srun -N 4 -n 4 -p skx-dev -A TG-PHY210106 -t 01:00:00 --pty /bin/bash -l


import glob
import netCDF4 as nc
import numpy as np
#import matplotlib.pyplot as plt

class ds:
    def __init__(self):
        self.x = []
        self.y = []
        self.v = []
        self.vx = []

#A function to extract data
def extract(filename,ds_out):
    fn = filename    
    ds = nc.Dataset(fn)      #importing the file
    #print(ds)                #checking datasets
    ds_out.filename = filename
    ds_out.lat  = ds['XLAT'][:]       #lattitude location [degree]
    ds_out.long = ds['XLONG'][:]      #longitude location [degree]
    ds_out.u    = ds['U'][:]          #x velocity [m/s]
    ds_out.v    = ds['V'][:]          #y velocity [m/s]
    ds_out.w    = ds['W'][:]          #z velocity [m/s]   
    ds_out.abs_vel= np.sqrt(ds_out.u[:,:,:,:-1]**2+ds_out.v[:,:,:-1,:]**2) #absolute velocity [m/s]
    ds_out.angle =  np.arctan2(ds_out.v[:,:,:-1,:],ds_out.u[:,:,:,:-1]) * 180 / np.pi + 180  #resultant velocity [degree]   
    return ds_out

'''
Folder = '/scratch/08457/tg877561/windstudy_PR/prWRF_2KM_2020_12/'

File = 'wrfout_d01_2020-12-01_00:00:00'

Data = nc.Dataset(Folder+File,"r")
'''

#list_of_paths = glob.glob('/scratch/08457/tg877561/windstudy_PR/prWRF_2KM_2020_12/wrfout*', recursive=True)
list_of_paths = glob.glob('./Data/NetCDF_data/wrfout*', recursive=True)

ds = extract(list_of_paths[0],ds)
ds.abs_vel_reshaped = (ds.abs_vel.data).reshape(*ds.abs_vel.shape[:1], -1)  
abs_vel_combined_reshaped = ds.abs_vel_reshaped.copy()

#ds = nc.Dataset(list_of_paths)

for file in list_of_paths[1:]:
    print(file)
    ds = extract(file,ds)
    ds.abs_vel_reshaped = (ds.abs_vel.data).reshape(*ds.abs_vel.shape[:1], -1)    
    
    abs_vel_combined_reshaped = np.vstack([abs_vel_combined_reshaped,ds.abs_vel_reshaped])    
    

abs_vel_mean     = abs_vel_combined_reshaped.mean(axis=0)

abs_vel_subtracted = abs_vel_combined_reshaped - np.repeat([abs_vel_mean],np.shape(abs_vel_combined_reshaped)[0],axis=0)    
 
# np.reshape(ds.abs_vel_subtracted,np.shape(ds.abs_vel))

#Performing reduced SVD
U, S, VT = np.linalg.svd(abs_vel_subtracted, full_matrices = False)

np.savez(f'Output.npz', U=U,S=S,VT=VT,abs_vel_mean=abs_vel_mean)

'''
j = 0

for r in (1,2,5,10,20):
    #Constructing a low rank approximation of the image
    Xapprox = U[:,:r] @ (np.diag(S[:r]) @ VT[:r,:])
    plt.figure(j+1)
    plt.contourf(ds.long[1,:,:],ds.lat[1,:,:] ,Xapprox.reshape(np.append(np.shape(abs_vel_combined_reshaped)[0],np.shape(ds.abs_vel)[1:4]))[0,0,:,:],cmap="coolwarm",levels=100)
    plt.title(f' r= {r}')
    clb=plt.colorbar()
    clb.set_label(r'Velocity (m/yr)')
    plt.savefig(f'svd_{r}modes_time0.pdf')
    j += 1

plt.figure(j+1)
plt.semilogy(S)
plt.ylabel('Singular values')
plt.xlabel('Number of modes')
plt.savefig('singular_values.jpg')

plt.figure(j+2)
plt.plot(np.cumsum((S))/np.sum((S)))
plt.ylabel('Cumulative sum of Singular values')
plt.xlabel('Number of modes')
plt.savefig('cum_sum_singular_values.pdf')


'''







