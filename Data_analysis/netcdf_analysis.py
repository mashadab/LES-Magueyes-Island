
"""
Analyzing the Wind velocity data
Mohammad Afzal Shadab
Date modified: 07/02/22
Email: mashadab@utexas.edu
"""

#Importing packages
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

#Making classes
class ds:
    def __init__(self):
        self.x = []
        self.y = []
        self.v = []
        self.vx = []

#A function to extract data
def extract(filename,ds_out):
    fn = filename + '.nc'    
    ds = nc.Dataset(fn)      #importing the file
    #print(ds)                #checking datasets
    ds_out.filename = filename
    ds_out.lat  = ds['XLAT'][:]       #x location [m]
    ds_out.long = ds['XLONG'][:]      #y location [m]
    ds_out.u    = ds['U'][:]          #absolute velocity [m/yr]
    ds_out.v    = ds['V'][:]         #x velocity [m/yr]
    ds_out.w    = ds['W'][:]         #x velocity [m/yr]    
    
    return ds_out

#A function to extract data
def plotting(ds_out):    
    #Absolute velocity error
    fig = plt.figure(figsize=(15,7.5) , dpi=100)
    plot = [plt.contourf(ds.long[1,:,:],ds.lat[1,:,:] ,np.sqrt(ds.u[1,1,:,:-1]**2+ds.v[1,1,:-1,:]**2),cmap="coolwarm",levels=100)]
    clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
    clb.set_label(r'Absolute Velocity (m/yr)')
    plt.ylabel(r'$y$')
    plt.xlabel(r'$x$')
    plt.axis('scaled')
    plt.savefig(f'{ds_out.filename}_2Dabs_velocity.pdf',bbox_inches='tight', dpi = 600)
 
    #Angle
    ds_out.angle =  np.arctan(ds.v[1,1,:-1,:]/ds.u[1,1,:,:-1]) * 180 / np.pi
    fig = plt.figure(figsize=(15,7.5) , dpi=100)
    plot = [plt.contourf(ds.long[1,:,:],ds.lat[1,:,:] ,(ds.angle),cmap="coolwarm",levels=100)]
    clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
    clb.set_label(r'Angle ($^o$)')
    plt.ylabel(r'$y$')
    plt.xlabel(r'$x$')
    plt.axis('scaled')
    plt.savefig(f'{ds_out.filename}_Angle.pdf',bbox_inches='tight', dpi = 600)
    
    #X velocity
    fig = plt.figure(figsize=(15,7.5) , dpi=100)
    plot = [plt.contourf(ds.long[1,:,:],ds.lat[1,:,:] ,(ds.u[1,1,:,:-1]),cmap="coolwarm",levels=100)]
    clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
    clb.set_label(r'X - Velocity (m/yr)')
    plt.ylabel(r'$y$')
    plt.xlabel(r'$x$')
    plt.axis('scaled')
    plt.savefig(f'{ds_out.filename}_Uvelocity.pdf',bbox_inches='tight', dpi = 600)
    
    #Y velocity
    fig = plt.figure(figsize=(15,7.5) , dpi=100)
    plot = [plt.contourf(ds.long[1,:,:],ds.lat[1,:,:] ,(ds.v[1,1,:-1,:]),cmap="coolwarm",levels=100)]
    clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
    clb.set_label(r'Y - Velocity (m/yr)')
    plt.ylabel(r'$y$')
    plt.xlabel(r'$x$')
    plt.axis('scaled')
    plt.savefig(f'{ds_out.filename}_Vvelocity.pdf',bbox_inches='tight', dpi = 600)
    
    return ds_out

#Extracting the data into a structure
ds = extract('./Data/wrfout_d01_2020-12-03_00_00_00',ds)

#Plotting the data
plotting(ds)




