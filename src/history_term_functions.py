"""
Functions used to calculate the history force based on the slip velocity
"""

import numpy as np 
import xarray as xr

def history_timescale(Rep, tau_diff):
    """"
    History timescale for Mei1992 history term with
    Dorgan2007 coeffients, sets timescale for time window
    needed when caluculate history term and depents on
    - tau_diff diffusion timescale particle (1/s)
    - Rep Particle Reynolds number
    """
    return tau_diff * (0.502/Rep + 0.123)**2

def Basset_kernel(t,s,d, nu):
    tau_diff = d * d / nu
    return (tau_diff / (4 * np.pi * (t-s)))**(0.5)

def Mei1992_kernel(t,s,c1,c2,d, nu, Rep):
    tau_diff = d * d / nu
    term1 = (4 * np.pi * (t-s) /tau_diff)**(1./(2 * c1))
    fh = (0.75 + c2 * Rep)**3
    term2 =  (Rep**3 * np.pi * (t-s)**2 / (fh * tau_diff**2) )**(1./c1) 
    return (term1+term2)**(-c1)


def Hinsberg(f, N, h):
    """
    Calculate history term using the expression proposed by Hinsberg 2011
    with N total number of segments
    h timestep
    f array of N elements (given by slip velocity or slip + part of Mei kernel)
    note f has length N+1 k in [0,N]
    """
    K0 = 4/3  * f[N] * np.sqrt(h)
    KN = f[0] * np.sqrt(h) * (N-4/3) / ((N-1)*np.sqrt(N - 1) +(N-3/2) * np.sqrt(N))
    KSUM = np.sum([np.sqrt(h) * f[N-k] * ( (k + 4/3) / ((k + 1)**(3/2) + (k + 3/2) * np.sqrt(k)) + (k - 4/3) /((k - 1)**(3/2) + (k - 3/2) * np.sqrt(k))) for k in range(1,N,1)])
    return K0 + KN + KSUM

def Daitche(f, N, h):
    """
    Calculate history term using the expression proposed by Daitche 2013
    with N total number of segments
    h timestep
    f array of N elements (given by slip velocity or slip + part of Mei kernel)
    note f has length N+1 k in [0,N]
    need quadruble (float128) precision for test case
    """
    K0 = (4/3  * np.sqrt(h,dtype = np.float128) * f[N] )
    KN =  4/3 * np.sqrt(h) * ((N - 1) * np.sqrt(N-1,dtype = np.float128) - N*np.sqrt(N,dtype = np.float128) + 6/4 * np.sqrt(N, dtype = np.float128)) * f[0] 
    KSUM = np.sum([4/3 * np.sqrt(h, dtype = np.float128) *((k-1)*np.sqrt(k-1,dtype = np.float128) + (k+1)*np.sqrt(k+1,dtype = np.float128) - 2 * k*np.sqrt(k,dtype = np.float128)) * f[N-k] for k in range(1,N,1)])
    return K0 + KN + KSUM


def f_Mei1992(dudt, t, c1, c2, Rep ,N ,h ,S):
    f_h = 0.75 + c2 * Rep
    f = [ dudt[k] * (1 + (np.sqrt(np.pi * (t -h * k)**3 / (S **3)) * Rep**3 / (2 * (f_h)**3 ))**(1/c1) )**(-c1) for k in range(0,N+1,1)]
    return f

def f_Mei1992_unit(dudt, t, c1, c2, Rep ,N ,h):
    # version where kernel is unitless and only slip velocity hase units, time intergral is in terms of t' =  4 pi t  /tau_diff 
    f_h = 0.75 + c2 * Rep
    f = [ dudt[k] * (1 + ((t -h * k)**(3/2) * Rep**3 / (16 * np.pi * (f_h)**3 ))**(1/c1) )**(-c1) for k in range(0,N+1,1)]
    return f


def History_Force_Hinsberg_Mei_kernel(dudt, t, c1, c2, Rep, N, dt,d,nu, rho):
    tau_diff = d * d / nu #diffusion timescale
    h = dt * 4 * np.pi / tau_diff
    tprime = t * 4 * np.pi / tau_diff
    f =  f_Mei1992_unit(dudt, tprime, c1, c2, Rep, N, h)
    history = tau_diff / (4 * np.pi) * Hinsberg(f,N,h)
    return 3*np.pi * d * nu * rho * history


def History_Force_Hinsberg_Basset_kernel(dudt, t, N, dt,d,nu, rho):
    tau_diff = d * d / nu #diffusion timescale
    h = dt * 4 * np.pi / tau_diff
    history = tau_diff / (4 * np.pi) * Hinsberg(dudt,N,h) 
    return 3 * np.pi * d * nu * rho * history

def History_time(tau_diff,Rep):
    return tau_diff * (0.502/Rep + 0.123)**2
