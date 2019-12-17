# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:46:56 2019

@author: dgbli

All landscape evolution models and figures for flat torus model with spectral
diffusion, presented at CSDMS in May 2019:

Litwin, D., C. J. Harman, T. Zaki, (2019): Implicit-spectral solution for a simple landscape evolution model. Poster. Community Surface Dynamics Modeling System Annual Meeting

"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

import LEMFunctions as lf
from landlab.components import LinearDiffuser, FlowAccumulator, FastscapeEroder
from landlab.plot import imshow_grid
from landlab import RasterModelGrid, CLOSED_BOUNDARY, FIXED_VALUE_BOUNDARY


TicToc = lf.TicTocGenerator()
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

#%%
#np.random.seed(1)
#Nc = 50
#N = int(Nc*3)
#dx = 10
#mg = RasterModelGrid((N,N), dx)
#z = mg.add_zeros('topographic__elevation', at='node' )
#mg.status_at_node[lf.get_fixed_cells(N)] = FIXED_VALUE_BOUNDARY
#
#imshow_grid(mg, mg.status_at_node, color_for_closed='blue')
##%% Flat torus, advective only
#
#z_orig = np.random.rand(Nc,Nc)
#z1 = np.vstack([z_orig, z_orig, z_orig])
#z_comp = np.hstack([z1, z1, z1])
#z_comp = z_comp.reshape((np.size(z_comp)))
#fixed = lf.get_fixed_cells(N)
#z_comp[fixed] = 0
#z[:] = z_comp
#
#imshow_grid(mg,'topographic__elevation')
#
#dt = 100
#T_max = 1E5
#nt = int(T_max//dt)
#D = 1E-4
#uplift_rate = 1E-3 #m/yr
#uplift_per_step = uplift_rate*dt
#m = 0.5 #Exponent on A []
#n = 1.0 #Exponent on S []
#K = 1E-11*(365*24*3600) #erosivity coefficient [yr−1]
#
#ld = LinearDiffuser(mg,linear_diffusivity=D)
#fr = FlowAccumulator(mg,'topographic__elevation',flow_director='D8')
#sp = FastscapeEroder(mg,K_sp = K,m_sp = m, n_sp=n)
#
#lf.tic()
#for i in range(nt):
#    fr.run_one_step()
#    sp.run_one_step(dt)
#
#    z_tiled_0 = z.reshape((mg.shape))
#    z_tiled_1 = lf.retile_array(z_tiled_0,Nc)
#    z_tiled_1 = z_tiled_1.reshape((np.size(z_tiled_1)))
#    z_tiled_1[fixed] = 0
#    z[:] = z_tiled_1
#
##    ld.run_one_step(dt)
#    mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_per_step
#
#    if i % 20 == 0:
#        print ('Completed loop %d' % i)
#
#lf.toc()
#imshow_grid(mg,'topographic__elevation')
#z_array_landlab = z.reshape((mg.shape))
#
#z_center = z_array_landlab[Nc:2*Nc,Nc:2*Nc]
#plt.imshow(z_array_landlab, cmap=plt.get_cmap('copper'))
#

#%% Vary N Flat torus, landlab diffuser
a = -500
b = 500
Nc_range = [24,48,96]
results_landlab = []


for j in range(len(Nc_range)):
    Nc = Nc_range[j]
    N = int(Nc*3)
    fixed = lf.get_fixed_cells(N)
    dx = (b-a)/N
    mg = RasterModelGrid((N,N), dx)
    z = mg.add_zeros('topographic__elevation', at='node' )
    mg.status_at_node[fixed] = FIXED_VALUE_BOUNDARY

#    imshow_grid(mg, mg.status_at_node, color_for_closed='blue')

    seed = np.random.seed(1)
    z_orig = np.random.rand(Nc,Nc)
    z1 = np.vstack([z_orig, z_orig, z_orig])
    z_comp = np.hstack([z1, z1, z1])
    z_comp = z_comp.reshape((np.size(z_comp)))
    z_comp[fixed] = 0
    z[:] = z_comp

#    imshow_grid(mg,'topographic__elevation')

    dt = 100
    T_max = 1E5
    nt = int(T_max//dt)
    D = 1E-3
    uplift_rate = 1E-3 #m/yr
    uplift_per_step = uplift_rate*dt
    m = 0.5 #Exponent on A []
    n = 1.0 #Exponent on S []
    K = 3E-4 #erosivity coefficient [yr−1]

    ld = LinearDiffuser(mg,linear_diffusivity=D)
    fr = FlowAccumulator(mg,'topographic__elevation',flow_director='D8')
    sp = FastscapeEroder(mg,K_sp = K,m_sp = m, n_sp=n)

    lf.tic()
    for i in range(nt):
        fr.run_one_step()
        sp.run_one_step(dt)
        ld.run_one_step(dt)

        z_tiled_0 = z.reshape((mg.shape))
        z_tiled_1 = lf.retile_array(z_tiled_0,Nc)
        z_tiled_1 = z_tiled_1.reshape((np.size(z_tiled_1)))
        z_tiled_1[fixed] = 0
        z[:] = z_tiled_1


        mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_per_step

        if i % 20 == 0:
            print ('Completed loop %d' % i)

    lf.toc()
#    imshow_grid(mg,'topographic__elevation')
    z_array = z.reshape((mg.shape))
#
#    z_center = z_array[Nc:2*Nc,Nc:2*Nc]
#    plt.imshow(z_array, cmap=plt.get_cmap('copper'))
#
#    pickle.dump(z_array, open('Landlab_torus_test_' + str(Nc) +'.p','wb'))

    results_landlab.append(z_array)

    del mg

pickle.dump(results_landlab, open('Landlab_torus_test_all_new.p','wb'))

#%% Vary N Flat torus, explicit spectral diffuser

a = -500
b = 500
Nc_range = [24,48,96]
results_exp_spectral = []

for j in range(len(Nc_range)):
    Nc = Nc_range[j]
    N = int(Nc*3)
    fixed = lf.get_fixed_cells(N)
    dx = (b-a)/N
    mg = RasterModelGrid((N,N), dx)
    z = mg.add_zeros('topographic__elevation', at='node' )
    mg.status_at_node[fixed] = FIXED_VALUE_BOUNDARY

#    imshow_grid(mg, mg.status_at_node, color_for_closed='blue')

    seed = np.random.seed(1)
    z_orig = np.random.rand(Nc,Nc)
    z1 = np.vstack([z_orig, z_orig, z_orig])
    z_comp = np.hstack([z1, z1, z1])
    z_comp = z_comp.reshape((np.size(z_comp)))
    z_comp[fixed] = 0
    z[:] = z_comp

#    imshow_grid(mg,'topographic__elevation')

    dt = 100
    T_max = 1E5
    nt = int(T_max//dt)
    D = 1E-3
    uplift_rate = 1E-3 #m/yr
    uplift_per_step = uplift_rate*dt
    m = 0.5 #Exponent on A []
    n = 1.0 #Exponent on S []
    K = 3E-4 #erosivity coefficient [yr−1]

    ld = LinearDiffuser(mg,linear_diffusivity=D)
    fr = FlowAccumulator(mg,'topographic__elevation',flow_director='D8')
    sp = FastscapeEroder(mg,K_sp = K,m_sp = m, n_sp=n)
    A = lf.Spectral_Diffuser(z_orig,D,dt,dx,dx,'explicit')

    lf.tic()
    for i in range(nt):
        fr.run_one_step()
        sp.run_one_step(dt)

        z_tiled_0 = z.reshape((mg.shape))
        z_center = z_tiled_0[Nc:2*Nc,Nc:2*Nc]
        z_center_1 = lf.Spectral_Diffuser_one_step(z_center,A)
        z_tiled_1 = lf.tile_array(z_center_1)
        z_tiled_1 = z_tiled_1.reshape((np.size(z_tiled_1)))
        z_tiled_1[fixed] = 0
        z[:] = z_tiled_1

    #    ld.run_one_step(dt)
        mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_per_step

        if i % 20 == 0:
            print ('Completed loop %d' % i)

    lf.toc()
#    imshow_grid(mg,'topographic__elevation')
    z_array = z.reshape((mg.shape))

#    pickle.dump(z_array, open('Landlab_exp_spectral_torus_test_' + str(Nc) +'.p','wb'))

    results_exp_spectral.append(z_array)

    del mg

pickle.dump(results_exp_spectral, open('Landlab_exp_spectral_torus_test_all_new.p','wb'))

#%% Vary N Flat torus, implicit spectral diffuser

a = -500
b = 500
Nc_range = [24,48,96]
results_imp_spectral = []

for j in range(len(Nc_range)):
    Nc = Nc_range[j]
    N = int(Nc*3)
    fixed = lf.get_fixed_cells(N)
    dx = (b-a)/N
    mg = RasterModelGrid((N,N), dx)
    z = mg.add_zeros('topographic__elevation', at='node' )
    mg.status_at_node[fixed] = FIXED_VALUE_BOUNDARY

#    imshow_grid(mg, mg.status_at_node, color_for_closed='blue')

    seed = np.random.seed(1)
    z_orig = np.random.rand(Nc,Nc)
    z1 = np.vstack([z_orig, z_orig, z_orig])
    z_comp = np.hstack([z1, z1, z1])
    z_comp = z_comp.reshape((np.size(z_comp)))
    z_comp[fixed] = 0
    z[:] = z_comp

#    imshow_grid(mg,'topographic__elevation')

    dt = 100
    T_max = 1E5
    nt = int(T_max//dt)
    D = 1E-3
    uplift_rate = 1E-3 #m/yr
    uplift_per_step = uplift_rate*dt
    m = 0.5 #Exponent on A []
    n = 1.0 #Exponent on S []
    K = 3E-4 #erosivity coefficient [yr−1]

    ld = LinearDiffuser(mg,linear_diffusivity=D)
    fr = FlowAccumulator(mg,'topographic__elevation',flow_director='D8')
    sp = FastscapeEroder(mg,K_sp = K,m_sp = m, n_sp=n)
    A = lf.Spectral_Diffuser(z_orig,D,dt,dx,dx,'implicit')

    lf.tic()
    for i in range(nt):
        fr.run_one_step()
        sp.run_one_step(dt)

        z_tiled_0 = z.reshape((mg.shape))
        z_center = z_tiled_0[Nc:2*Nc,Nc:2*Nc]
        z_center_1 = lf.Spectral_Diffuser_one_step(z_center,A)
        z_tiled_1 = lf.tile_array(z_center_1)
        z_tiled_1 = z_tiled_1.reshape((np.size(z_tiled_1)))
        z_tiled_1[fixed] = 0
        z[:] = z_tiled_1

    #    ld.run_one_step(dt)
        mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_per_step

        if i % 20 == 0:
            print ('Completed loop %d' % i)

    lf.toc()
#    imshow_grid(mg,'topographic__elevation')
    z_array = z.reshape((mg.shape))

#    pickle.dump(z_array, open('Landlab_imp_spectral_torus_test_' + str(Nc) +'.p','wb'))

    results_imp_spectral.append(z_array)

    del mg

pickle.dump(results_imp_spectral, open('Landlab_imp_spectral_torus_test_all_new.p','wb'))




#%% Vary dt Flat torus, landlab diffuser
a = -500
b = 500
Nc = 64
N = int(Nc*3)
fixed = lf.get_fixed_cells(N)
dx = (b-a)/N
dt_range = [10,50,100,500,1000,2000]
results_landlab = []


for j in range(len(dt_range)):

    mg = RasterModelGrid((N,N), dx)
    z = mg.add_zeros('topographic__elevation', at='node' )
    mg.status_at_node[fixed] = FIXED_VALUE_BOUNDARY

#    imshow_grid(mg, mg.status_at_node, color_for_closed='blue')

    seed = np.random.seed(1)
    z_orig = np.random.rand(Nc,Nc)
    z1 = np.vstack([z_orig, z_orig, z_orig])
    z_comp = np.hstack([z1, z1, z1])
    z_comp = z_comp.reshape((np.size(z_comp)))
    z_comp[fixed] = 0
    z[:] = z_comp

#    imshow_grid(mg,'topographic__elevation')

    dt = dt_range[j]
    T_max = 1E5
    nt = int(T_max//dt)
    D = 1E-3
    uplift_rate = 1E-3 #m/yr
    uplift_per_step = uplift_rate*dt
    m = 0.5 #Exponent on A []
    n = 1.0 #Exponent on S []
    K = 3E-4 #erosivity coefficient [yr−1]

    ld = LinearDiffuser(mg,linear_diffusivity=D)
    fr = FlowAccumulator(mg,'topographic__elevation',flow_director='D8')
    sp = FastscapeEroder(mg,K_sp = K,m_sp = m, n_sp=n)

    lf.tic()
    for i in range(nt):
        fr.run_one_step()
        sp.run_one_step(dt)
        ld.run_one_step(dt)

        z_tiled_0 = z.reshape((mg.shape))
        z_tiled_1 = lf.retile_array(z_tiled_0,Nc)
        z_tiled_1 = z_tiled_1.reshape((np.size(z_tiled_1)))
        z_tiled_1[fixed] = 0
        z[:] = z_tiled_1


        mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_per_step

        if i % 100 == 0:
            print ('Completed loop %d' % i)

    lf.toc()
#    imshow_grid(mg,'topographic__elevation')
    z_array = z.reshape((mg.shape))
#
#    z_center = z_array[Nc:2*Nc,Nc:2*Nc]
#    plt.imshow(z_array, cmap=plt.get_cmap('copper'))
#
#    pickle.dump(z_array, open('Landlab_torus_test_' + str(Nc) +'.p','wb'))

    results_landlab.append(z_array)

    del mg

pickle.dump(results_landlab, open('Landlab_torus_test_dt_all.p','wb'))

#%% Vary dt Flat torus, explicit spectral diffuser

a = -500
b = 500
Nc = 64
N = int(Nc*3)
fixed = lf.get_fixed_cells(N)
dx = (b-a)/N
dt_range = [10,50,100,500,1000,2000]
results_exp_spectral = []

for j in range(len(dt_range)):

    mg = RasterModelGrid((N,N), dx)
    z = mg.add_zeros('topographic__elevation', at='node' )
    mg.status_at_node[fixed] = FIXED_VALUE_BOUNDARY

#    imshow_grid(mg, mg.status_at_node, color_for_closed='blue')

    seed = np.random.seed(1)
    z_orig = np.random.rand(Nc,Nc)
    z1 = np.vstack([z_orig, z_orig, z_orig])
    z_comp = np.hstack([z1, z1, z1])
    z_comp = z_comp.reshape((np.size(z_comp)))
    z_comp[fixed] = 0
    z[:] = z_comp

#    imshow_grid(mg,'topographic__elevation')

    dt = dt_range[j]
    T_max = 1E5
    nt = int(T_max//dt)
    D = 1E-3
    uplift_rate = 1E-3 #m/yr
    uplift_per_step = uplift_rate*dt
    m = 0.5 #Exponent on A []
    n = 1.0 #Exponent on S []
    K = 3E-4 #erosivity coefficient [yr−1]

    ld = LinearDiffuser(mg,linear_diffusivity=D)
    fr = FlowAccumulator(mg,'topographic__elevation',flow_director='D8')
    sp = FastscapeEroder(mg,K_sp = K,m_sp = m, n_sp=n)
    A = lf.Spectral_Diffuser(z_orig,D,dt,dx,dx,'explicit')

    lf.tic()
    for i in range(nt):
        fr.run_one_step()
        sp.run_one_step(dt)

        z_tiled_0 = z.reshape((mg.shape))
        z_center = z_tiled_0[Nc:2*Nc,Nc:2*Nc]
        z_center_1 = lf.Spectral_Diffuser_one_step(z_center,A)
        z_tiled_1 = lf.tile_array(z_center_1)
        z_tiled_1 = z_tiled_1.reshape((np.size(z_tiled_1)))
        z_tiled_1[fixed] = 0
        z[:] = z_tiled_1

    #    ld.run_one_step(dt)
        mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_per_step

        if i % 100 == 0:
            print ('Completed loop %d' % i)

    lf.toc()
#    imshow_grid(mg,'topographic__elevation')
    z_array = z.reshape((mg.shape))

#    pickle.dump(z_array, open('Landlab_exp_spectral_torus_test_' + str(Nc) +'.p','wb'))

    results_exp_spectral.append(z_array)

    del mg

pickle.dump(results_exp_spectral, open('Landlab_exp_spectral_torus_test_dt_all.p','wb'))

#%% Vary dt, Flat torus, implicit spectral diffuser

a = -500
b = 500
Nc = 64
N = int(Nc*3)
fixed = lf.get_fixed_cells(N)
dx = (b-a)/N
dt_range = [10,50,100,500,1000,2000]
results_imp_spectral = []

for j in range(len(dt_range)):

    mg = RasterModelGrid((N,N), dx)
    z = mg.add_zeros('topographic__elevation', at='node' )
    mg.status_at_node[fixed] = FIXED_VALUE_BOUNDARY

#    imshow_grid(mg, mg.status_at_node, color_for_closed='blue')

    seed = np.random.seed(1)
    z_orig = np.random.rand(Nc,Nc)
    z1 = np.vstack([z_orig, z_orig, z_orig])
    z_comp = np.hstack([z1, z1, z1])
    z_comp = z_comp.reshape((np.size(z_comp)))
    z_comp[fixed] = 0
    z[:] = z_comp

#    imshow_grid(mg,'topographic__elevation')

    dt = dt_range[j]
    T_max = 1E5
    nt = int(T_max//dt)
    D = 1E-3
    uplift_rate = 1E-3 #m/yr
    uplift_per_step = uplift_rate*dt
    m = 0.5 #Exponent on A []
    n = 1.0 #Exponent on S []
    K = 3E-4 #erosivity coefficient [yr−1]

    ld = LinearDiffuser(mg,linear_diffusivity=D)
    fr = FlowAccumulator(mg,'topographic__elevation',flow_director='D8')
    sp = FastscapeEroder(mg,K_sp = K,m_sp = m, n_sp=n)
    A = lf.Spectral_Diffuser(z_orig,D,dt,dx,dx,'implicit')

    lf.tic()
    for i in range(nt):
        fr.run_one_step()
        sp.run_one_step(dt)

        z_tiled_0 = z.reshape((mg.shape))
        z_center = z_tiled_0[Nc:2*Nc,Nc:2*Nc]
        z_center_1 = lf.Spectral_Diffuser_one_step(z_center,A)
        z_tiled_1 = lf.tile_array(z_center_1)
        z_tiled_1 = z_tiled_1.reshape((np.size(z_tiled_1)))
        z_tiled_1[fixed] = 0
        z[:] = z_tiled_1

    #    ld.run_one_step(dt)
        mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_per_step

        if i % 100 == 0:
            print ('Completed loop %d' % i)

    lf.toc()
#    imshow_grid(mg,'topographic__elevation')
    z_array = z.reshape((mg.shape))

#    pickle.dump(z_array, open('Landlab_imp_spectral_torus_test_' + str(Nc) +'.p','wb'))

    results_imp_spectral.append(z_array)

    del mg

pickle.dump(results_imp_spectral, open('Landlab_imp_spectral_torus_test_dt_all.p','wb'))



#%% flat torus, spectral diffuser, 3 point BCs:

a = -500
b = 500


Nc = 128
N = int(Nc*3)
dx = (b-a)/N
mg = RasterModelGrid((N,N), dx)
z = mg.add_zeros('topographic__elevation', at='node' )
fixed = lf.get_fixed_cells_3(N)
mg.status_at_node[fixed] = FIXED_VALUE_BOUNDARY

#imshow_grid(mg, mg.status_at_node, color_for_closed='blue')

seed = np.random.seed(1)
z_orig = np.random.rand(Nc,Nc)
z1 = np.vstack([z_orig, z_orig, z_orig])
z_comp = np.hstack([z1, z1, z1])
z_comp = z_comp.reshape((np.size(z_comp)))
z_comp[fixed] = 0
z[:] = z_comp

dt = 100
T_max = 1E5
nt = int(T_max//dt)
D = 1E-3
uplift_rate = 1E-3 #m/yr
uplift_per_step = uplift_rate*dt
m = 0.5 #Exponent on A []
n = 1.0 #Exponent on S []
K = 3E-4 #erosivity coefficient [yr−1]

fr = FlowAccumulator(mg,'topographic__elevation',flow_director='D8')
sp = FastscapeEroder(mg,K_sp = K,m_sp = m, n_sp=n)
A = lf.Spectral_Diffuser(z_orig,D,dt,dx,dx,'implicit')

lf.tic()
for i in range(nt):
    fr.run_one_step()
    sp.run_one_step(dt)

    z_tiled_0 = z.reshape((mg.shape))
    z_center = z_tiled_0[Nc:2*Nc,Nc:2*Nc]
    z_center_1 = lf.Spectral_Diffuser_one_step(z_center,A)
    z_tiled_1 = lf.tile_array(z_center_1)
    z_tiled_1 = z_tiled_1.reshape((np.size(z_tiled_1)))
    z_tiled_1[fixed] = 0 #boundary condition
    z[:] = z_tiled_1

#    ld.run_one_step(dt)
    mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_per_step

    if i % 20 == 0:
        print ('Completed loop %d' % i)

lf.toc()
imshow_grid(mg,'topographic__elevation')
z_array_3_bcs = z.reshape((mg.shape))


pickle.dump(z_array_3_bcs, open('Landlab_imp_spectral_3bcs_2' + str(Nc) +'.p','wb'))



#%%
#
#plt.figure()
#plt.imshow(results_imp_spectral[2])
#plt.title('Implicit Spectral Diffuser')
#plt.colorbar()
#
#plt.figure()
#plt.imshow(results_landlab[0])
#plt.title('Landlab Diffuser')
#plt.colorbar()
#
#%% Figure comparing landlab, explicit spectral, implicit spectral

#find max and min
all_images = results_landlab+results_exp_spectral+results_imp_spectral
max_all = 0
for array in all_images:
    max1 = np.max(array)
    if max1>max_all:
        max_all = max1

fig, axs = plt.subplots(nrows=3, ncols=3,figsize=[10,10])
plt.tight_layout()
for i in range(0,3):
    im1 = axs[i,0].imshow(all_images[i*2], vmin=0, vmax=max_all, extent=[0,1000,0,1000])
    im2 = axs[i,1].imshow(all_images[i*2+6], vmin=0, vmax=max_all, extent=[0,1000,0,1000])
    im3 = axs[i,2].imshow(all_images[i*2+12], vmin=0, vmax=max_all, extent=[0,1000,0,1000])

axs[0,0].xaxis.set_ticklabels([])
axs[0,1].axes.yaxis.set_ticklabels([])
axs[0,1].axes.xaxis.set_ticklabels([])
axs[0,2].axes.yaxis.set_ticklabels([])
axs[0,2].axes.xaxis.set_ticklabels([])

axs[1,0].xaxis.set_ticklabels([])
axs[1,1].axes.yaxis.set_ticklabels([])
axs[1,1].axes.xaxis.set_ticklabels([])
axs[1,2].axes.yaxis.set_ticklabels([])
axs[1,2].axes.xaxis.set_ticklabels([])

axs[2,1].axes.yaxis.set_ticklabels([])
axs[2,2].axes.yaxis.set_ticklabels([])

axs[2,1].set_xlabel('Distance (m)')
axs[1,0].set_ylabel('Distance (m)')

fig.subplots_adjust(right=0.8,left=0.3,bottom=0.3,wspace=0.05,hspace=0.1)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cb = fig.colorbar(im1, cax=cbar_ax)
cb.set_label('Elevation (m)')


#%% Mean elevation plot: change dt

results_imp_spectral = pickle.load(open('Landlab_imp_spectral_torus_test_dt_all.p',"rb"))
results_exp_spectral = pickle.load(open('Landlab_exp_spectral_torus_test_dt_all.p',"rb"))
results_landlab = pickle.load(open('Landlab_torus_test_dt_all.p',"rb"))
all_images = results_landlab+results_exp_spectral+results_imp_spectral
dt_range = [10,50,100,500,1000,2000]

mean_elevations = np.zeros(len(all_images))
for i in range(len(all_images)):
    mean_elevations[i] = np.mean(all_images[i],axis=None)

mean_elevations = np.reshape(mean_elevations,(6,3),'F')

dt_range_plt = np.flip(dt_range)
mean_elevations_plt = np.flipud(mean_elevations)

plt.figure(figsize=[6,4])
plt.plot(dt_range_plt,mean_elevations_plt[:,0],color=colors['burlywood'],label='Landlab')
plt.plot(dt_range_plt,mean_elevations_plt[:,1],color=colors['orchid'],label='Explicit Spectral')
plt.plot(dt_range_plt,mean_elevations_plt[:,2],color=colors['mediumslateblue'],label='Implicit Spectral')
plt.xticks(dt_range_plt)
plt.xlabel('Timestep [yr]')
plt.ylabel('Mean Elevation [m]')
plt.xscale('log')
plt.legend(frameon = False)
plt.tight_layout()

#%% mean elevation plot: change N

#results_imp_spectral = pickle.load(open('Landlab_imp_spectral_torus_test_all_new.p',"rb"))
#results_exp_spectral = pickle.load(open('Landlab_exp_spectral_torus_test_all_new.p',"rb"))
#results_landlab = pickle.load(open('Landlab_torus_test_all_new.p',"rb"))

results_imp_spectral_1 = pickle.load(open('Landlab_imp_spectral_torus_test_all.p',"rb"))
results_exp_spectral_1 = pickle.load(open('Landlab_exp_spectral_torus_test_all.p',"rb"))
results_landlab_1 = pickle.load(open('Landlab_torus_test_all.p',"rb"))

results_N = [24,48,96]
results_N_1 = [32,64,128]

results_N_all = results_N + results_N_1
results_imp_spectral_all = results_imp_spectral+results_imp_spectral_1
results_exp_spectral_all = results_exp_spectral+results_exp_spectral_1
results_landlab_all = results_landlab+results_landlab_1
all_images = results_landlab_all+results_exp_spectral_all+results_imp_spectral_all

mean_elevations = np.zeros(len(all_images))
for i in range(len(all_images)):
    mean_elevations[i] = np.mean(all_images[i],axis=None)

mean_elevations = np.reshape(mean_elevations,(6,3),'F')
index_sorted = np.argsort(np.array(results_N_all))
for j in range(3):
    mean_elevations[:,j] = mean_elevations[index_sorted,j]
results_N_all_sort = np.sort(np.array(results_N_all))

plt.figure(figsize=[6,4])
plt.plot(results_N_all_sort,mean_elevations[:,0],color=colors['burlywood'],label='Landlab')
plt.plot(results_N_all_sort,mean_elevations[:,1],color=colors['orchid'],label='Explicit Spectral')
plt.plot(results_N_all_sort,mean_elevations[:,2],color=colors['mediumslateblue'],label='Implicit Spectral')
plt.xticks(results_N_all_sort)
plt.xlabel('Number of grid cells N')
plt.ylabel('Mean Elevation [m]')
#plt.xscale('log')
plt.legend(frameon = False)
plt.tight_layout()


#
##%%
#
#
#eta0 = pickle.load(open('Landlab_imp_spectral_switch_bcs_2_0.p',"rb"))
#eta1 = pickle.load(open('Landlab_imp_spectral_switch_bcs_2_50.p',"rb"))
#eta2 = pickle.load(open('Landlab_imp_spectral_switch_bcs_2_100.p',"rb"))
#eta3 = pickle.load(open('Landlab_imp_spectral_switch_bcs_2_150.p',"rb"))
#eta4 = pickle.load(open('Landlab_imp_spectral_switch_bcs_2_200.p',"rb"))
#
##plt.figure()
##plt.imshow(eta0)
#
##find max and min
#all_images = [eta0,eta1,eta2,eta3,eta4]
#max_all = 0
#for array in all_images:
#    max1 = np.max(array)
#    if max1>max_all:
#        max_all = max1
##
##fig, axs = plt.subplots(nrows=1, ncols=5)
##fig.tight_layout()
##for i in range(0,5):
##    im1 = axs[i].imshow(all_images[i], vmin=0, vmax=max_all)
##
##fig.subplots_adjust(right=0.8)
##cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
##fig.colorbar(im1, cax=cbar_ax)
#
#
#for i in range(0,5):
#    fig,ax = plt.subplots()
#    im = ax.imshow(all_images[i], vmin=0, vmax=max_all)
#    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#    fig.colorbar(im1, cax=cbar_ax)
#    plt.savefig('Drainage_capture_'+str(i)+'.png')
#

#%% voronoi polygons
from scipy.spatial import Voronoi, voronoi_plot_2d
Nc = 128
N = Nc*3
a = -500
b = 500
dx = (b-a)/N

n_array = np.zeros((N**2))
point_nums = lf.get_fixed_cells_3(N)
n_array[point_nums] = 1
n_array = n_array.reshape((N,N)).T
indices = np.where(n_array==1)
x = dx*indices[0].reshape((len(indices[0]),1))
y = dx*indices[1].reshape((len(indices[0]),1))
pairs = np.concatenate((x,y),axis=1)


vor = Voronoi(pairs)
poly = voronoi_plot_2d(vor)
plt.imshow(z_array_3_bcs,origin='lower',extent=[0,1000,0,1000])
fig = plt.gcf()
fig.set_size_inches(10.5,10.5)
plt.show()
plt.xlim(0,1000)
plt.ylim(0,1000)
plt.savefig('Voronoi_3bcs.png', bbox_inches = 'tight', pad_inches = 0)

#plt.figure()
#imshow_grid(mg, mg.status_at_node, color_for_closed='blue')
