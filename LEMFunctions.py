# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 09:02:01 2019

@author: dgbli


Functions needed to run explicit finite difference, explicit and implicit spectral
diffusion in Landlab. Also included are the functions needed to create the flat
torus seen in the poster:

Litwin, D., C. J. Harman, T. Zaki, (2019): Implicit-spectral solution for a simple landscape evolution model. Poster. Community Surface Dynamics Modeling System Annual Meeting

"""

import numpy as np
import time

def Explicit_Diffuser_one_step(eta,D,dt,dy,dx):
    """
    Explicit central difference diffusion with a 3-point stencil in each dimension.
    Does not operate on boundary (Assumed dirichlet eta = 0)

    eta = elevation
    D = scalar linear diffusivity
    dt = timestep
    dx, dy = raster grid spacing

    """

    eta_1 = np.zeros_like(eta)
    Nx = np.shape(eta)[0]
    Ny = np.shape(eta)[1]
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            eta_1[i,j] = eta[i,j] + D*dt/dx**2*(eta[i-1,j]-2*eta[i,j]+eta[i+1,j]) + D*dt/dy**2*(eta[i,j-1]-2*eta[i,j]+eta[i,j-1])
    return eta_1


def Explicit_Diffuser_looped_one_step(eta,D,dt,dy,dx):
    """
    Explicit central difference diffusion with a 3-point stencil in each dimension.
    Periodic boundary condition top-bottom and left-right

    eta = elevation
    D = scalar linear diffusivity
    dt = timestep
    dx, dy = raster grid spacing

    """
    eta_1 = np.zeros_like(eta)
    Nx = np.shape(eta)[0]
    Ny = np.shape(eta)[1]
    for i in range(0,Nx):
        for j in range(0,Ny):
            eta_1[i,j] = eta[i,j] + D*dt/dx**2*(eta[i-1,j]-2*eta[i,j]+eta[np.mod(i+1,Nx),j]) + D*dt/dy**2*(eta[i,j-1]-2*eta[i,j]+eta[i,np.mod(j+1,Ny)])
    return eta_1

def Spectral_Diffuser(eta,D,dt,dy,dx,method):
    """
    2D linear diffusion using a spectral method with periodic boundary conditions.
    This method generates the update matrix A.

    eta = elevation
    D = scalar linear diffusivity
    dt = timestep
    dx, dy = raster grid spacing
    method: 'explicit' or 'implicit'.

    """

    #calculations for wavenumbers kx and ky.
    Nx = np.shape(eta)[0]
    Ny = np.shape(eta)[1]

    nkx = np.zeros(Nx)
    nkx[0:int(Nx/2)] = np.arange(0,Nx/2)
    nkx[int(Nx/2):] = np.arange(-Nx/2,0)
    kx = 2*np.pi/(dx*Nx)*nkx

    nky = np.zeros(Ny)
    nky[0:int(Ny/2)] = np.arange(0,Ny/2)
    nky[int(Ny/2):] = np.arange(-Ny/2,0)
    ky = 2*np.pi/(dy*Ny)*nky

    kX2, kY2 = np.meshgrid(kx**2,ky**2)

    #create update matrix based on selected method
    if method=='implicit':
        A = np.divide((1-(D*dt/2)*kX2-(D*dt/2)*kY2), (1+(D*dt/2)*kX2+(D*dt/2)*kY2))
    elif method == 'explicit':
        A = (1-D*dt*kX2 - D*dt*kY2)
    else:
        return 'Choose implicit or explicit'
    return A

def Spectral_Diffuser_one_step(eta,A):
    """
    Run the spectral diffuser one timestep, as defined in matrix A.
    Note that boundary conditions are periodic top-bottom and left-right.

    eta = elevation
    A = update matrix

    """

    #convert elevation to fourier domain
    eta_ft = np.fft.fft2(eta)

    #update
    eta_ft_1 = np.multiply(A,eta_ft)

    #convert back to spatial domain
    eta = np.real(np.fft.ifft2(eta_ft_1))

    return eta

# following functions create a 9-tile domain from the topography, which is
# only necessary to accomodate the lack of periodic boundary conditions
# implemented in the FastscapeEroder. For examples of what this looks like,
# see included poster.

def retile_array(z_tiled_0,Nc):
    """
    Take 9-tile array z_tiled_0, with single tile dimension NcxNc.
    Extract center tile and duplicate it to create new 9-tile array.
    """
    z_center = z_tiled_0[Nc:2*Nc,Nc:2*Nc]
    zc = np.vstack([z_center, z_center, z_center])
    z_tiled_1 = np.hstack([zc, zc, zc])
    return z_tiled_1

def tile_array(z_center):
    """
    Take array z_center and duplicate it to create 9-tile array.
    """
    zc = np.vstack([z_center, z_center, z_center])
    z_tiled_1 = np.hstack([zc, zc, zc])
    return z_tiled_1

def get_fixed_cells(N):
    """
    To create a flat torous, place fixed cells in the interior of the tiles.
    Given total dimension of 9-tiled array N, this function returns the indices
    of the flattened array that should be fixed value. See ./figures/poster.
    """

    cells = np.zeros(9,dtype=int)
    cells[0] = N*(N//6)+1*N//6
    cells[1] = N*(N//6)+3*N//6
    cells[2] = N*(N//6)+5*N//6

    cells[3] = N*(N//2)+1*N//6
    cells[4] = N*(N//2)+3*N//6
    cells[5] = N*(N//2)+5*N//6

    cells[6] = N*(N//(6/5))+1*N//6
    cells[7] = N*(N//(6/5))+3*N//6
    cells[8] = N*(N//(6/5))+5*N//6

    return cells

def get_fixed_cells_3(N):
    """
    To create a flat torous, place fixed cells in the interior of the tiles.
    Given total dimension of 9-tiled array N, this function returns the indices
    of the flattened array that should be fixed value. 3 fixed points in each
    tile.
    """

    Nc = N//3
    center = np.zeros((Nc,Nc),dtype=int)

    center[Nc//5,Nc//5] = 1
    center[Nc//5,4*Nc//5] = 1
    center[2*Nc//3,Nc//3] = 1

    full = tile_array(center)
    full = full.reshape((np.size(full),1))
    cells = np.where(full==1)[0]
    return cells

def get_fixed_cells_1(N):
    """
    To create a flat torous, place fixed cells in the interior of the tiles.
    Given total dimension of 9-tiled array N, this function returns the indices
    of the flattened array that should be fixed value. 1 fixed point in each
    tile.
    """

    Nc = N//3
    center = np.zeros((Nc,Nc),dtype=int)
    center[2*Nc//3,Nc//3] = 1

    full = tile_array(center)
    full = full.reshape((np.size(full),1))
    cells = np.where(full==1)[0]
    return cells


def TDMA(a,b,c,d):
    """
    Solves a tridiagonal matrix using the Thomas algoithm.

    a = lower off-diagonal
    b = diagonal
    c = upper off-diagonal
    d = RHS
    used for the implicit in y, spectral in x method.

    """

    n = len(d)

    if type(d[0]) == np.complex128:
        cp= np.zeros(n-1,dtype=complex)
        dp= np.zeros(n,dtype=complex)
        x = np.zeros(n,dtype=complex)
    else:
        cp= np.zeros(n-1,dtype=float)
        dp= np.zeros(n,dtype=float)
        x = np.zeros(n,dtype=float)

    cp[0] = c[0]/b[0]
    dp[0] = d[0]/b[0]

    for i in range(1,n-1):
        cp[i] = c[i]/(b[i] - a[i-1]*cp[i-1])
    for i in range(1,n):
        dp[i] = (d[i] - a[i-1]*dp[i-1])/(b[i] - a[i-1]*cp[i-1])
    x[n-1] = dp[n-1]
    for i in range(n-1,0,-1):
        x[i-1] = dp[i-1] - cp[i-1]*x[i]
    return x


##### This function is old and doesn't work. The wavenumber is incorrect. It solves the diffusion equation spectal in x, implicit in y
def Implicit_Spectral_Diffuser_one_step(eta,D,dt,dy,dx):

   """
   Implicit in the y-direction, spectral in the x-direction
   Crank-Nicholson forumulation in time
   implemented with dirichlet boundary conditions at y=0 and y=L (eta=0)
   Periodic boundary conditions at x=0 and x=L

   eta = elevation
   D = scalar linear diffusivity
   dt = timestep
   dx, dy = raster grid spacing

   """

   #modified wavenumber
   Nx = np.shape(eta)[0]
   Ny = np.shape(eta)[1]
   f = np.fft.rfftfreq(Nx,d=dx) #frequency
   k = 2*np.pi*f #wavenumber
   kp = np.sqrt((2*(1-np.cos(k*dx)))/dx**2) #modified wavenumber

   #matrix coefficients
   a = -D*dt/(2*dy**2)
   c = -D*dt/(2*dy**2)
   A = np.concatenate((a*np.ones(Ny-2,dtype=complex),[0]))
   C = np.concatenate(([0],c*np.ones(Ny-2,dtype=complex)))

   b_kp = 1 + (D*dt*kp**2)/2 + D*dt/dy**2
#    b_k = 1+(D*dt*k**2)/2+D*dt/dy**2

   #initialize
   eta_ft = np.fft.rfft(eta,n=None,axis=0)
   eta_1_ft = np.zeros_like(eta_ft,dtype=complex)

   for n in range(len(f)):
       rhs = np.zeros(Ny,dtype=complex)
       for j in range(1,Ny-1):
           rhs[j] = D*dt/(2*dy**2)*eta_ft[n,j-1] + (1 - D*dt*kp[n]**2/2 - D*dt/dy**2)*eta_ft[n,j] + D*dt/(2*dy**2)*eta_ft[n,j+1]
       B = np.concatenate(([1],b_kp[n]*np.ones(Ny-2,dtype=complex),[1]))
       eta_1_ft[n,:] = TDMA(A,B,C,rhs)

   eta = np.fft.irfft(eta_1_ft, n=None, axis=0)

   return eta


# matlab like tic toc. Why? just because.
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
