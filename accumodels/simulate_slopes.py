#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import scipy as sp
import matplotlib as mpl
mpl.use("TkAgg")
mpl.rcParams['pdf.fonttype'] = 42

import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from IPython import embed as shell

from sim_tools import get_DDM_traces, apply_bounds_diff_trace, _bounds, _bounds_collapse_linear, _bounds_collapse_hyperbolic
from sim_tools import summary_plot, conditional_response_plot
from tqdm import tqdm

sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 0.25, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 6, 
    'ytick.labelsize': 6, 
    'legend.fontsize': 6, 
    'xtick.major.width': 0.25, 
    'ytick.major.width': 0.25,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()

def do_simulations(params):
    rt = []
    response = []
    stimulus = []
    traces = []
    for stim in [1,0]:
        
        # get traces:
        x = get_DDM_traces(v=params['v'],
                            z=params['z'],
                            dc=params['dc'],
                            dc_slope=params['dc_slope'],
                            sv=params['sv'],
                            stim=stim,
                            nr_trials=params['nr_trials'],
                            tmax=tmax,
                            dt=dt,)
        
        # get bounds:
        if params['bound'] == 'default':
            b1, b0 = _bounds(a=params['a'], tmax=tmax, dt=dt)
        elif params['bound'] == 'collapse_linear':
            b1, b0 = _bounds_collapse_linear(a=params['a'], c1=params['c1'], c0=params['c0'], tmax=tmax, dt=dt)
        elif params['bound'] == 'collapse_hyperbolic':
            b1, b0 = _bounds_collapse_hyperbolic(a=params['a'], c=params['c'], tmax=tmax, dt=dt)
        
        # apply bounds:
        rt_dum, response_dum = apply_bounds_diff_trace(x=x, b1=b1, b0=b0)
        
        # store results:
        rt.append((rt_dum*dt)+ndt)
        response.append(response_dum)
        stimulus.append(np.ones(params['nr_trials']) * stim)
        traces.append(x)

    df = pd.DataFrame()
    df.loc[:,'rt'] = np.concatenate(rt)
    df.loc[:,'response'] = np.concatenate(response)
    df.loc[:,'stimulus'] = np.concatenate(stimulus)
    df.loc[:,'correct'] = np.array(np.concatenate(stimulus) == np.concatenate(response), dtype=int)
    df.loc[:,'subj_idx'] = params['subj_idx']
    df.to_csv(os.path.join(data_folder, 'df_{}.csv'.format(params['subj_idx'])))

    traces = np.vstack(traces)
    for i in range(traces.shape[0]):
        if sum(traces[i,:]>params['a']) > 0:
            traces[i,np.where(traces[i,:]>params['a'])[0][0]:] = params['a']
        if sum(traces[i,:]<0) > 0:
            traces[i,np.where(traces[i,:]<0)[0][0]:] = 0
    
    hit = np.array((df['stimulus']==1)&(df['response']==1))
    fa = np.array((df['stimulus']==0)&(df['response']==1))
    miss = np.array((df['stimulus']==1)&(df['response']==0))
    cr = np.array((df['stimulus']==0)&(df['response']==0))

    shell()

    fig = plt.figure(figsize=(2,2))

    for t in traces[hit,:][0:5000]:
        plt.plot(t, alpha=0.005, lw=1, color='black')
    # for t in traces[fa,:][0:500]:
    #     plt.plot(t, alpha=0.02, color='orange')
    # for t in traces[miss,:][0:500]:
    #     plt.plot(t, alpha=0.02, color='green')
    # for t in traces[cr,:][0:500]:
    #     plt.plot(t, alpha=0.02, color='green')

    for trial, color, alpha in zip([hit, fa, miss, cr], ['orange', 'orange', 'green', 'green'], [1,0.5,0.5,1]):
        y = np.nanmean(traces[trial,:], axis=0)
        x = np.arange(y.shape[0])
        ind = np.zeros(y.shape[0], dtype=bool)
        ind[0:20] = True
        (m,b) = sp.polyfit(x[ind], y[ind], 1)
        regression_line = sp.polyval([m,b],x)
        plt.plot(y, color=color, lw=2, alpha=alpha)
        plt.plot(x[ind], regression_line[ind], color='black', lw=1, ls='--')
        print(m)
    plt.axhline(0, color='k', lw=0.5)
    plt.axhline(params['a'], color='k', lw=0.5)
    plt.axhline(params['z'], color='k', lw=0.5)
    plt.xlim(0,30)
    plt.ylim(-0.1,params['a']+0.1)
    plt.xlabel('Timesteps')
    plt.ylabel('Decision variable')
    sns.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(fig_folder, 'simulate_slopes_{}.pdf'.format(params['subj_idx'])))

data_folder = os.path.expanduser('~/projects/2018_Urai_choice-history-ddm/ddm_data/')
fig_folder = os.path.expanduser('~/projects/2018_Urai_choice-history-ddm/ddm_figs/')
fits_folder = os.path.expanduser('~/projects/2018_Urai_choice-history-ddm/fits/')

simulate = True
parallel = False
# nr_trials = int(1e5) #100K
nr_trials = int(1e4) #10.000
tmax = 5
dt = 0.01

v = 1
a = 1
dc = 0
dc_slope = 0
ndt = 0.1
sv = 0.5

sArray = [

    {'subj_idx':17, 'v':v, 'dc':dc+0.6, 'z':0.5*a, 'a':a, 'dc_slope':dc_slope, 'sv':sv, 'bound':'default', 'nr_trials':nr_trials},

    ]

if simulate:
	if not parallel:
		for i, s in tqdm(enumerate(sArray)):
			do_simulations(s) 
	else:
	    from joblib import Parallel, delayed
	    n_jobs = 42
	    res = Parallel(n_jobs=n_jobs)(delayed(do_simulations)(params) for params in sArray)
	    # do_simulations(sArray[0])