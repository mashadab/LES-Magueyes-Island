
"""
Analyzing the Wind velocity data using Spectral POD
Mohammad Afzal Shadab
Date modified: 07/12/22
Email: mashadab@utexas.edu
Website: https://github.com/tjburrows/spod_python
"""

#Importing packages
import netCDF4 as nc
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
    ds_out.lat  = ds['XLAT'][:]       #lattitude location [degree]
    ds_out.long = ds['XLONG'][:]      #longitude location [degree]
    ds_out.u    = ds['U'][:]          #x velocity [m/s]
    ds_out.v    = ds['V'][:]          #y velocity [m/s]
    ds_out.w    = ds['W'][:]          #z velocity [m/s]   
    ds_out.abs_vel= np.sqrt(ds_out.u[:,:,:,:-1]**2+ds_out.v[:,:,:-1,:]**2) #absolute velocity [m/s]
    ds_out.angle =  np.arctan2(ds_out.v[:,:,:-1,:],ds_out.u[:,:,:,:-1]) * 180 / np.pi + 180  #resultant velocity [degree]   
    return ds_out

#A function to extract data
def plotting(ds_out):    
    #Absolute velocity
    fig = plt.figure(figsize=(15,7.5) , dpi=100)
    plot = [plt.contourf(ds.long[1,:,:],ds.lat[1,:,:],ds_out.abs_vel[1,1,:,:],cmap="coolwarm",levels=100)]
    clb = fig.colorbar(plot[0], orientation='vertical',aspect=50, pad=-0.1)
    clb.set_label(r'Absolute Velocity (m/yr)')
    plt.ylabel(r'$y$')
    plt.xlabel(r'$x$')
    plt.axis('scaled')
    plt.savefig(f'{ds_out.filename}_2Dabs_velocity.pdf',bbox_inches='tight', dpi = 600)
 
    #Angle
    fig = plt.figure(figsize=(15,7.5) , dpi=100)
    plot = [plt.contourf(ds.long[1,:,:],ds.lat[1,:,:] ,(ds.angle[1,1,:,:]),cmap="coolwarm",levels=100)]
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
#plotting(ds)


#Reshape the data except for first value which is time in hours
ds.abs_vel_reshaped = ds.abs_vel.reshape(*ds.abs_vel.shape[:1], -1)
ds.abs_vel_mean     = ds.abs_vel_reshaped.mean(axis=0)

ds.abs_vel_subtracted = ds.abs_vel_reshaped - np.repeat([ds.abs_vel_mean],np.shape(ds.abs_vel)[0],axis=0)    
 
# np.reshape(ds.abs_vel_subtracted,np.shape(ds.abs_vel))

#Performing reduced SVD
U, S, VT = np.linalg.svd(ds.abs_vel_subtracted, full_matrices = False)

j = 0

for r in (1,2):
    #Constructing a low rank approximation of the image
    Xapprox = U[:,:r] @ (np.diag(S[:r]) @ VT[:r,:])
    plt.figure(j+1)
    img = plt.imshow(Xapprox)
    plt.set_cmap('gray')
    plt.axis('off')
    plt.title(f' r= {r}')
    plt.savefig(f'svd_{r}modes.jpg')
    j += 1

plt.figure(j+1)
plt.semilogy(np.diag(S))
plt.ylabel('Singular values')
plt.xlabel('Number of modes')
plt.savefig('singular_values.jpg')

plt.figure(j+2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.ylabel('Cumulative sum of Singular values')
plt.xlabel('Number of modes')
plt.savefig('cum_sum_singular_values.jpg')


