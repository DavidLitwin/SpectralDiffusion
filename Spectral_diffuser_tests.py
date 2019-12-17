# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:11:22 2019

@author: dgbli

Some comparisons with analytical solutions

"""
import numpy as np
import matplotlib.pyplot as plt
import dill
from LEMFunctions import Spectral_Diffuser, Spectral_Diffuser_one_step, Explicit_Diffuser_one_step, Explicit_Diffuser_looped_one_step

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


#true 1D solution:
eta_true = lambda t,x,D: 1/np.sqrt(4*np.pi*D*t)*np.exp(-x**2/(4*D*t))

#true 2D solution:
eta_true_2 = lambda t,x,y,D: 1/np.sqrt(4*np.pi*D*t)**2*np.exp(-x**2/(4*D*t)-y**2/(4*D*t))


#%% Make 1D diffuser and compare with true solution for dirac delta
N_range = np.arange(50,1150,100)
peaks_est = np.zeros(len(N_range))
peaks_true = np.zeros(len(N_range))
error = np.zeros(len(N_range))
mass = np.zeros(len(N_range))
mass_true = np.zeros(len(N_range))
mass_init = np.zeros(len(N_range))

for j in range(len(N_range)):
    N = N_range[j]
    eta = np.zeros(N)
    eta[N//2] = 1
    a = -50
    b = 50 #4*np.pi
    D = 1E-4
    dt = 10
    T_max = 1E4
    n = T_max//dt
    dx = (b-a)/N
    x = np.arange(a,b,dx)
    nk = np.zeros(N)
    nk[0:int(N/2)] = np.arange(0,N/2)
    nk[int(N/2):] = np.arange(-N/2,0)

    k = 2*np.pi/(N*dx)*nk
    #kp = np.sqrt((2*(1-np.cos(k*dx)))/dx**2)


    for i in range(int(n)):
        eta_fft = np.fft.fft(eta)
        eta_1_fft = (1 - D*dt*k**2)*eta_fft

        eta = np.real(np.fft.ifft(eta_1_fft))

    eta_t = 1*dx*eta_true(dt*n,x,D)

    peaks_est[j] = np.max(eta)
    peaks_true[j] = np.max(eta_t)
    error[j] = rmse(eta,eta_t)
    mass[j] = dx*np.sum(eta)
    mass_true[j] = dx*np.sum(eta_t)
    mass_init[j] = 1*dx
#    plt.figure()
#    plt.plot(x,np.real(eta))
#    plt.plot(x,eta_t)
#    plt.xlim((-100,100))
#    plt.ylim((0,1))
#    plt.savefig('1D_diffuse_test_N_'+str(N)+'_dt_'+str(dt)+'.png')

plt.figure()
plt.plot(N_range,peaks_est, label='Numerical')
plt.plot(N_range,peaks_true, label='True')
plt.xlabel('Number of points N')
plt.ylabel('Peak value after T=1E4, dt=1')
plt.legend()
#plt.savefig('Peaks_dt_1.png')

plt.figure()
plt.plot(N_range,error)
plt.xlabel('Number of points N')
plt.ylabel('RMSE after T=1E4, dt=1')
#plt.savefig('Error_dt_1.png')


plt.figure()
plt.plot(N_range,mass)
plt.plot(N_range,mass_true)
plt.scatter(N_range,mass_init)
plt.xlabel('Number of points N')
plt.ylabel('Mass after T=1E4, dt=1')
#plt.savefig('mass_dt_10.png')


#%% implicit and explicit spectral test

N = 100
dt = 500
a = -500
b = 500 #4*np.pi
D = 1E-4
dt = 100
T_max = 1E6
n = T_max//dt
dx = (b-a)/N
x = np.arange(a,b,dx)
y = np.arange(a,b,dx)

X,Y = np.meshgrid(x,y)
eta_imp = np.zeros((N,N))
eta_imp[N//2,N//2] = 1
eta_exp = eta_imp[:]
eta_exp_fd = eta_imp[:]

A_imp = Spectral_Diffuser(eta_imp,D,dt,dx,dx,'implicit')
A_exp = Spectral_Diffuser(eta_exp,D,dt,dx,dx,'explicit')


for i in range(int(n)):
    eta_imp = Spectral_Diffuser_one_step(eta_imp,A_imp)
    eta_exp = Spectral_Diffuser_one_step(eta_exp,A_exp)
    eta = Explicit_Diffuser_one_step(eta,D,dt,dx,dx)

    if i % 100 == 0:
        print ('Completed loop %d' % i)

#true:
eta_t_2 = 1*dx**2*eta_true_2(dt*n,X,Y,D)

eta_exp_sort = np.flipud(np.sort(eta_exp,axis=None))
eta_imp_sort = np.flipud(np.sort(eta_imp,axis=None))
eta_t_sort = np.flipud(np.sort(eta_t_2,axis=None))

plt.figure()
plt.plot(eta_exp_sort, '-',label = 'explicit')
plt.plot(eta_imp_sort, '--',label = 'implicit')
plt.plot(eta_t_sort, label = 'true')
plt.legend()


plt.figure()
plt.imshow(eta_imp)
#%% test effects of varying N
#N_range = np.array([50,80,100,150,200,300,400,500,1000])
N_range = 2**np.arange(5,11)
a = -500
b = 500 #4*np.pi
D = 1E-3
dt = 2500
T_max = 1E6
n = T_max//dt

error_imp = np.zeros(len(N_range))
error_exp = np.zeros(len(N_range))
error_exp_fd = np.zeros(len(N_range))
mass_exp = np.zeros(len(N_range))
mass_exp_fd = np.zeros(len(N_range))
mass_imp = np.zeros(len(N_range))
mass_true = np.zeros(len(N_range))
mass_init = np.zeros(len(N_range))

true_results = []
exp_results = []
exp_fd_results = []
imp_results = []


for j in range(len(N_range)):
    N = N_range[j]
    dx = (b-a)/N
    x = np.arange(a,b,dx)
    y = np.arange(a,b,dx)
    X,Y = np.meshgrid(x,y)

    eta_imp = np.zeros((N,N))
    eta_imp[N//2,N//2] = 1
    eta_exp = eta_imp[:]
    eta_exp_fd = eta_imp[:]


    A_imp = Spectral_Diffuser(eta_imp,D,dt,dx,dx,'implicit')
    A_exp = Spectral_Diffuser(eta_exp,D,dt,dx,dx,'explicit')

    for i in range(int(n)):
        eta_imp = Spectral_Diffuser_one_step(eta_imp,A_imp)
        eta_exp = Spectral_Diffuser_one_step(eta_exp,A_exp)
        eta_exp_fd = Explicit_Diffuser_looped_one_step(eta_exp_fd,D,dt,dx,dx)

        if i % 100 == 0:
            print ('Completed loop %d' % i)

    eta_t = 1*dx**2*eta_true_2(dt*n,X,Y,D)

    error_imp[j] = rmse(eta_imp,eta_t)
    error_exp[j] = rmse(eta_exp,eta_t)
    error_exp_fd[j] = rmse(eta_exp_fd,eta_t)
    mass_exp[j] = dx**2*np.sum(eta_exp)
    mass_exp_fd[j] = dx**2*np.sum(eta_exp)
    mass_imp[j] = dx**2*np.sum(eta_imp)
    mass_true[j] = dx**2*np.sum(eta_t)
    mass_init[j] = 1*dx**2

    true_results.append(eta_t)
    imp_results.append(eta_imp)
    exp_results.append(eta_exp)
    exp_fd_results.append(eta_exp)


dill.dump_session('Spectral_N_range_session_dt_2500_D_1E-3.pkl')

dill.load_session('Spectral_N_range_session_dt_2500_D_1E-3.pkl')
#%% Plots

#example of the diffusion after 1E6 yrs, 1km^2 domain, D = 1E-3, dt=2500
result = true_results[3]
plt.figure(figsize=(6,6))
im = plt.imshow(result,extent=[0,b-a,0,b-a])
cb = plt.colorbar(im,format='%.0e',fraction=0.046, pad=0.04)
cb.set_label('Mass')
plt.tight_layout()


#%% mass plot

#mass plot dt=2500
N_plot = N_range

mass_exp_fd_plot = mass_exp_fd[:]
mass_exp_fd_plot[mass_exp_fd_plot<0] = 0
mass_exp_fd_plot[np.isnan(mass_exp_fd_plot)] = 0

mass_exp_plot = mass_exp[:]
mass_exp_plot[mass_exp_plot<0] = 0
mass_exp_plot[np.isnan(mass_exp_plot)] = 0

#plt.figure()
#plt.plot(N_plot,mass_true,linewidth=3, label='True')
#plt.plot(N_plot,mass_imp,'--',linewidth=3,label='Implicit Spectral')
#plt.plot(N_plot,mass_exp_plot,linewidth=3,label='Explicit Spectral')
#plt.plot(N_plot,mass_exp_fd_plot,'--',linewidth=3,label='Explicit Finite Difference')
#plt.legend()


#%% Mass bar chart

n_groups = 6

fig,ax = plt.subplots(figsize=[6,4])
index = np.arange(n_groups)
bar_width = 0.15

plt_true_mass = plt.bar(index,mass_true, bar_width, color = 'gray', label='True')
plt_ll_mass = plt.bar(index+bar_width,mass_l, bar_width, color = 'peru', label='Landlab')
plt_imp_mass = plt.bar(index+2*bar_width,mass_imp, bar_width, color = 'slateblue', label='Implicit Spectral')
plt_exp_mass = plt.bar(index+3*bar_width,mass_exp_plot, bar_width, color = 'dodgerblue', label='Explicit Spectral')
ax.set_yscale('log')
ax.axes.xaxis.set_ticks([])

plt.xlabel('Grid point side dimension N')
plt.ylabel('Mass')
plt.xticks(index + bar_width, ('32', '64', '128', '256','512','1048'))
plt.legend(loc='northeast')

#%% Landlab diffusion test
from landlab.components import LinearDiffuser
from landlab.plot import imshow_grid
from landlab import RasterModelGrid

results_l = []
N_range = 2**np.arange(5,11)
a = -500
b = 500 #4*np.pi
D = 1E-3
dt = 2500
T_max = 1E6
n = T_max//dt


for j in range(len(N_range)):
    N = N_range[j]
    dx = (b-a)/N
    x = np.arange(a,b,dx)
    y = np.arange(a,b,dx)
    X,Y = np.meshgrid(x,y)

    eta_l = np.zeros((N,N))
    eta_l[N//2,N//2] = 1
    eta_l = eta_l.reshape((np.size(eta_l)))

    mg = RasterModelGrid((N,N), dx)
    z = mg.add_zeros('topographic__elevation', at='node' )
    z[:] = eta_l

    ld = LinearDiffuser(mg,linear_diffusivity=D)

    for i in range(int(n)):
        ld.run_one_step(dt)

        if i % 100 == 0:
           print ('Completed loop %d' % i)

    z_array = z.reshape((mg.shape))
    results_l.append(z_array)

    del mg

mass_l = []
for i in range(6):
    N = N_range[i]
    dx = (b-a)/N
    mass_l.append(dx**2*np.sum(results_l[i]))

#%%

dill.dump_session('Spectral_N_range_session_dt_2500_D_1E-3.pkl')
