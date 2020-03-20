#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
from IPython import embed as shell

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

def get_DDM_traces(v=1, z=0.5, dc=0, dc_slope=0, sv=0.1, noise_sd=1, stim=0, nr_trials=1000, tmax=5.0, dt=0.01):
    
    """
    DDM

    v:  mean drift rate
    z:  starting point
    dc: drift criterion
    """
    
    if stim == 0:
        v = np.random.normal(-v,sv,nr_trials)
    elif stim == 1:
        v = np.random.normal(v,sv,nr_trials)
    x = np.zeros((nr_trials, int(tmax/dt)))
    x[:,:] = np.NaN
    x[:,0] = z
    for i in range((int(tmax/dt))-1):
        x[:,i+1] = x[:,i] + ((v + dc + (dc_slope*dt*i) ) * dt) + (np.random.normal(0,noise_sd,nr_trials)*np.sqrt(dt))
    return x
    
def get_OU_traces(v, ll, dc, z, noise_sd=1, pre_generated=False, stim=0, nr_trials=1000, tmax=5.0, dt=0.01):
    
    """
    OU-model
    
    v:  mean drift rate
    ll: Ornstein-Uhlenbeck process parameter (effective leak / self-excitation)
    z:  starting point
    dc: drift criterion
    """
    
    if stim == 0:
        v = v[::-1]
    
    x1 = np.zeros((nr_trials, int(tmax/dt)))
    x2 = np.zeros((nr_trials, int(tmax/dt)))
    x1[:,:] = np.NaN
    x2[:,:] = np.NaN
    x1[:,0] = z[0]
    x2[:,0] = z[1]
    for i in range((int(tmax/dt))-1):
        if pre_generated:
            x1[:,i+1] = x1[:,i] + v[0][:,i] + dc[0] - (ll[0]*x1[:,i])
            x2[:,i+1] = x2[:,i] + v[1][:,i] + dc[1] - (ll[1]*x2[:,i])
        else:
            x1[:,i+1] = x1[:,i] + ((v[0] + dc[0] - (ll[0]*x1[:,i])) * dt) + (np.random.normal(0,noise_sd/np.sqrt(2),nr_trials)*np.sqrt(dt))
            x2[:,i+1] = x2[:,i] + ((v[1] + dc[1] - (ll[1]*x2[:,i])) * dt) + (np.random.normal(0,noise_sd/np.sqrt(2),nr_trials)*np.sqrt(dt))
    return x1-x2

def get_LCA_traces(v, k, w, dc, z, noise_sd=1, pre_generated=False, stim=0, nr_trials=1000, tmax=5.0, dt=0.01):
    
    """
    LCA
    """
    
    if stim == 0:
        v = v[::-1]
    
    x1 = np.zeros((nr_trials, int(tmax/dt)))
    x2 = np.zeros((nr_trials, int(tmax/dt)))
    x1[:,:] = np.NaN
    x2[:,:] = np.NaN
    x1[:,0] = z[0]
    x2[:,0] = z[1]
    for i in range((int(tmax/dt))-1):
        if pre_generated:
            x1[:,i+1] = np.clip(x1[:,i] + v[0][:,i] + dc[0] - (k[0]*x1[:,i]) - (w[1]*x2[:,i]), a_min=0, a_max=1e6)
            x2[:,i+1] = np.clip(x2[:,i] + v[1][:,i] + dc[1] - (k[1]*x2[:,i]) - (w[0]*x1[:,i]), a_min=0, a_max=1e6)
        else:
            x1[:,i+1] = np.clip(x1[:,i] + ((v[0] + dc[0] - (k[0]*x1[:,i]) - (w[1]*x2[:,i])) * dt) + (np.random.normal(0,noise_sd,nr_trials)*np.sqrt(dt)), a_min=0, a_max=1e6)
            x2[:,i+1] = np.clip(x2[:,i] + ((v[1] + dc[1] - (k[1]*x2[:,i]) - (w[0]*x1[:,i])) * dt) + (np.random.normal(0,noise_sd,nr_trials)*np.sqrt(dt)), a_min=0, a_max=1e6)
    return x1, x2

def _bounds(a, lower_is_0=True, tmax=5, dt=0.01):
    t = np.arange(0, tmax, dt)
    b1 = np.ones(len(t)) * a
    if lower_is_0:
        b0 = np.zeros(len(t))
    else:
        b0 = -b1
    return b1, b0

def _bounds_collapse_linear(a, c1, c0, lower_is_0=True, tmax=5, dt=0.01):
    t = np.arange(0, tmax, dt)
    b1 = (a)-(c1*t)
    if lower_is_0:
        b0 = 0+(c0*t)
    else:
        b0 = -b1
    return b1, b0

def _bounds_collapse_hyperbolic(a, c, lower_is_0=True, tmax=5, dt=0.01):

    t = np.arange(0, tmax, dt)
    b1 = (a)-a*(t/(t+c))
    if lower_is_0:
        b0 = -b1+a
    else:
        b0 = -b1
    
    return b1, b0 

def apply_bounds_diff_trace(x, b1, b0):
    rt = np.zeros(x.shape[0])
    response = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j] >= b1[j]:
                rt[i] = j
                response[i] = 1
                break
            elif x[i,j] <= b0[j]:
                rt[i] = j
                response[i] = 0
                break
    return rt, response

def apply_bounds_accumulater_traces(x1, x2, a=[0.15, 0.15],):
    
    rt = np.zeros(x1.shape[0])
    response = np.zeros(x1.shape[0])
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            if x1[i,j] >= a[0]:
                rt[i] = j
                response[i] = 1
                break
            elif x2[i,j] >= a[1]:
                rt[i] = j
                response[i] = 0
                break
    return rt, response

def apply_timepoint_accumulater_traces(x1, x2, timepoint=None):
    
    if timepoint:
        response = np.array(x1[:,timepoint] > x2[:,timepoint], dtype=int)
        rt = np.ones(x1.shape[0]) * timepoint
    else:
        response = np.array(x1[:,-1] > x2[:,-1], dtype=int)
        rt = np.ones(x1.shape[0]) * x1.shape[1]
    
    return rt, response