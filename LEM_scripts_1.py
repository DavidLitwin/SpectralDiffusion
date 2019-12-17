# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:38:14 2019

@author: dgbli

Run spectral diffuser in a landscape evolution model. This code does not account
for a difference in boundary conditions between the diffusion model (periodic)
and the FastscapeEroder (fixed gradient). This is the reason why the tiled approach
that appears in Flat_torus_LEM was used.

"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

import LEMFunctions as lf
from landlab.components import LinearDiffuser, FlowAccumulator, FastscapeEroder
from landlab.plot import imshow_grid
from landlab import RasterModelGrid, CLOSED_BOUNDARY, FIXED_VALUE_BOUNDARY
from matplotlib.pyplot import figure, show, plot, xlabel, ylabel, title

TicToc = lf.TicTocGenerator()

#%% Run with landlab diffusion
np.random.seed(3)
Nx = 100
Ny = 100
dx = 10
mg = RasterModelGrid((Nx,Ny), dx)
z = mg.add_zeros('topographic__elevation', at='node' )
#mg.core_nodes['topographic__elevation'] = np.random.rand(mg.number_of_core_nodes)
mg.set_status_at_node_on_edges(right=FIXED_VALUE_BOUNDARY, top=FIXED_VALUE_BOUNDARY,left=FIXED_VALUE_BOUNDARY,bottom=FIXED_VALUE_BOUNDARY)

z_array = z.reshape((mg.shape))
z_init = np.zeros(mg.shape)
z_init[1:-1,1:-1] = np.random.rand(np.shape(z_array[1:-1,1:-1])[0],np.shape(z_array[1:-1,1:-1])[1])
z_init = z_init.reshape((np.size(z_init)))
z[:] = z_init

imshow_grid(mg,'topographic__elevation')

dt = 50
T_max = 1E5
nt = int(T_max//dt)
D = 1E-4
uplift_rate = 1E-3 #m/yr
uplift_per_step = uplift_rate*dt
m = 0.5 #Exponent on A []
n = 1.0 #Exponent on S []
K = 1E-11*(365*24*3600) #erosivity coefficient [yr−1]

ld = LinearDiffuser(mg,linear_diffusivity=D)
fr = FlowAccumulator(mg,'topographic__elevation',flow_director='D8')
sp = FastscapeEroder(mg,K_sp = K,m_sp = m, n_sp=n)

lf.tic()
for i in range(nt):
    fr.run_one_step()
    sp.run_one_step(dt)
    ld.run_one_step(dt)
    mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_per_step

    if i % 20 == 0:
        print ('Completed loop %d' % i)

lf.toc()
imshow_grid(mg,'topographic__elevation')
z_array_landlab = z.reshape((mg.shape))

pickle.dump(z_array_landlab, open("Landlab_test2.p","wb"))

#%% Run with my explicit finite difference diffuser
mg1 = RasterModelGrid((Nx,Ny), dx)
z1 = mg1.add_zeros('topographic__elevation', at='node' )

mg1.set_status_at_node_on_edges(right=FIXED_VALUE_BOUNDARY, top=FIXED_VALUE_BOUNDARY,left=FIXED_VALUE_BOUNDARY,bottom=FIXED_VALUE_BOUNDARY)

dt = 50
T_max = 1E5
nt = int(T_max//dt)
D = 1E-4
uplift_rate = 1E-3 #m/yr
uplift_per_step = uplift_rate*dt
m = 0.5 #Exponent on A []
n = 1.0 #Exponent on S []
K = 1E-11*(365*24*3600) #erosivity coefficient [yr−1]

np.random.seed(3)
z_array = z1.reshape((mg1.shape))
z_init = np.zeros(mg1.shape)
z_init[1:-1,1:-1] = np.random.rand(np.shape(z_array[1:-1,1:-1])[0],np.shape(z_array[1:-1,1:-1])[1])
z_init = z_init.reshape((np.size(z_init)))
z1[:] = z_init #z is tied to the RasterModelGrid mg1

fr1 = FlowAccumulator(mg1,'topographic__elevation',flow_director='D8')
sp1 = FastscapeEroder(mg1,K_sp = K,m_sp = m, n_sp=n)

lf.tic()
for i in range(nt):
    fr1.run_one_step()
    sp1.run_one_step(dt)

    z_array = z1.reshape((mg1.shape))
    z_new = lf.Explicit_Diffuser_one_step(z_array,D,dt,dx,dx)
    z_new = z_new.reshape((np.size(z_new)))
    z1[:] = z_new

    mg1.at_node['topographic__elevation'][mg1.core_nodes] += uplift_per_step

    if i % 20 == 0:
        print ('Completed loop %d' % i)
lf.toc()

imshow_grid(mg1,'topographic__elevation')
z_array_explicit = z1.reshape((mg.shape))

pickle.dump(z_array_explicit, open("Landlab_explicit_test1.p","wb"))


#%% Run with full spectral diffuser
mg2 = RasterModelGrid((Nx,Ny), dx)
z2 = mg2.add_zeros('topographic__elevation', at='node' )

mg2.set_status_at_node_on_edges(right=FIXED_VALUE_BOUNDARY, top=FIXED_VALUE_BOUNDARY,left=FIXED_VALUE_BOUNDARY,bottom=FIXED_VALUE_BOUNDARY)


dt = 50
T_max = 1E5
nt = int(T_max//dt)
D = 1E-4
uplift_rate = 1E-3 #m/yr
uplift_per_step = uplift_rate*dt
m = 0.5 #Exponent on A []
n = 1.0 #Exponent on S []
K = 1E-11*(365*24*3600) #erosivity coefficient [yr−1]

np.random.seed(3)
z_array = z2.reshape((mg2.shape))
z_init = np.zeros(mg2.shape)
z_init[1:-1,1:-1] = np.random.rand(np.shape(z_array[1:-1,1:-1])[0],np.shape(z_array[1:-1,1:-1])[1])
z_init = z_init.reshape((np.size(z_init)))
z2[:] = z_init #z is tied to the RasterModelGrid mg2

fr1 = FlowAccumulator(mg2,'topographic__elevation',flow_director='D8')
sp1 = FastscapeEroder(mg2,K_sp = K,m_sp = m, n_sp=n)
A = lf.Spectral_Diffuser(z_array,D,dt,dx,dx,'explicit')

lf.tic()

for i in range(nt):
    fr1.run_one_step()
    sp1.run_one_step(dt)

    z_array = z2.reshape((mg2.shape))
    z_new = lf.Full_Spectral_Diffuser_one_step(z_array,A)
    z_new = z_new.reshape((np.size(z_new)))
    z2[:] = z_new

    mg2.at_node['topographic__elevation'][mg2.core_nodes] += uplift_per_step

    if i % 20 == 0:
        print ('Completed loop %d' % i)

lf.toc()

imshow_grid(mg2,'topographic__elevation')
z_array = z2.reshape((mg.shape))
pickle.dump(z_array, open("Landlab_full_spectral_test1.p","wb"))

#%%
z_array_spectral = pickle.load(open("Landlab_full_spectral_test1.p","rb"))

z_diff = z_array_landlab - z_array_spectral

plt.imshow(z_diff)
