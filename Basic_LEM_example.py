# -*- coding: utf-8 -*-
"""
Created on Sun May 26 17:47:27 2019

@author: dgbli

Experimenting with a simple landscape evolution model in Landlab.
Nothing spectral here.

"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

from landlab.components import LinearDiffuser, FlowAccumulator, FastscapeEroder
from landlab.plot import imshow_grid
from landlab import RasterModelGrid, CLOSED_BOUNDARY, FIXED_VALUE_BOUNDARY

import pickle
#%%

#grid
N = 50
dx = 10
mg = RasterModelGrid((N,N), dx)
z = mg.add_zeros('topographic__elevation', at='node' )
mg.set_status_at_node_on_edges(right=CLOSED_BOUNDARY, top=CLOSED_BOUNDARY, \
                              left=CLOSED_BOUNDARY, bottom=FIXED_VALUE_BOUNDARY)

#initial condition
z[:] = np.random.rand(len(z))

imshow_grid(mg,'topographic__elevation')

#imshow_grid(mg, mg.status_at_node, color_for_closed='blue')

#time info
dt = 50
T_max = 1E5
nt = int(T_max//dt)

#parameters
D = 1E-4
uplift_rate = 1E-3 #m/yr
uplift_per_step = uplift_rate*dt
m = 0.5 #Exponent on A []
n = 1.0 #Exponent on S []
K = 1E-11*(365*24*3600) #erosivity coefficient [yr−1]

ld = LinearDiffuser(mg,linear_diffusivity=D)
fr = FlowAccumulator(mg,'topographic__elevation',flow_director='D8')
sp = FastscapeEroder(mg,K_sp = K,m_sp = m, n_sp=n)

for i in range(nt):
    fr.run_one_step()
    sp.run_one_step(dt)
    ld.run_one_step(dt)

    mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_per_step

    if i % 50 == 0:
        print ('Completed loop %d' % i)

        fig = imshow_grid(mg,'topographic__elevation')
        plt.savefig('figure'+str(i)+'.png')
        plt.close()
