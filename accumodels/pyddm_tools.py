import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from joblib import Parallel, delayed

import ddm
from ddm import Sample
from ddm import plot
from ddm import models
from ddm import Model, Fittable, Fitted, Bound, Overlay, Solution
from ddm.functions import fit_adjust_model, display_model
from ddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture, OverlayUniformMixture, InitialCondition, ICPoint, ICPointSourceCenter, LossBIC
from ddm import set_N_cpus

from IPython import embed as shell

set_N_cpus(16)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'axes.linewidth': 0.25,
    'xtick.major.width': 0.25,
    'ytick.major.width': 0.25,
    'ytick.major.width': 0.25,
    'ytick.major.width': 0.25,
    'ytick.major.pad' : 2.0,
    'ytick.minor.pad' : 2.0,
    'xtick.major.pad' : 2.0,
    'xtick.minor.pad' : 2.0,
    'axes.labelpad' : 4.0,
    'axes.titlepad' : 6.0,
    } )
sns.plotting_context()

def make_z(n_bins, z_bin=False):
    if z_bin:
       z_names = ['z{}'.format(i) for i in range(n_bins)]
    else:
        z_names = ['z']
    class StartingPoint(InitialCondition):
        name = "A starting point."
        required_parameters = z_names
        if z_bin:
            required_conditions = ["bin"]
        def get_IC(self, x, dx, conditions):
            pdf = np.zeros(len(x))
            if z_bin:
                z_param = getattr(self, 'z{}'.format(conditions['bin']))
            else:
                z_param = self.z
            pdf[int(len(pdf)*z_param)] = 1
            return pdf
    return StartingPoint

def make_drift(n_bins, v_bin=False, dc_bin=False):
    if v_bin:
        v_names = ['v{}'.format(i) for i in range(n_bins)]
    else:
        v_names = ['v']
    if dc_bin:
        dc_names = ['b{}'.format(i) for i in range(n_bins)]
    else:
        dc_names = ['b']
    class DriftStimulusCoding(ddm.models.Drift):
        name = "Drift"
        required_parameters = v_names + dc_names
        if v_bin or dc_bin:
            required_conditions = ["stimulus", "bin"]
        else:
            required_conditions = ["stimulus"]
        def get_drift(self, conditions, **kwargs):
            if v_bin:
                v_param = getattr(self, 'v{}'.format(conditions['bin']))
            else:
                v_param = self.v
            if dc_bin:
                dc_param = getattr(self, 'b{}'.format(conditions['bin']))
            else:
                dc_param = self.b
            return dc_param + (v_param * conditions['stimulus'])
    return DriftStimulusCoding

def make_a(n_bins, a_bin=False, bound_collapse=False):
    if a_bin:
        a_names = ['a{}'.format(i) for i in range(n_bins)]
        if bound_collapse:
            c_names = ['c{}'.format(i) for i in range(n_bins)]
        else:
            c_names = []
    else:
        a_names = ['a']
        if bound_collapse:
            c_names = ['c']
        else:
            c_names = []
    class BoundCollapsingHyperbolic(Bound):
        name = "Hyperbolic collapsing bounds"
        required_parameters = a_names + c_names
        if a_bin:
            required_conditions = ["bin"]
        def get_bound(self, t, conditions, **kwargs):
            if a_bin:
                a_param = getattr(self, 'a{}'.format(conditions['bin']))
                if bound_collapse:
                    c_param = getattr(self, 'c{}'.format(conditions['bin']))
            else:
                a_param = self.a
                if bound_collapse:
                    c_param = self.c
            if bound_collapse:
                return a_param-(a_param*(t/(t+c_param)))
            else:
                return a_param
    return BoundCollapsingHyperbolic

def make_t(n_bins, t_bin=False):
    if t_bin:
       t_names = ['t{}'.format(i) for i in range(n_bins)]
    else:
        t_names = ['t']
    class NonDecisionTime(Overlay):
        name = "Non-decision time"
        required_parameters = t_names
        if t_bin:
            required_conditions = ["bin"]
        def apply(self, solution):
            # Unpack solution object
            corr = solution.corr
            err = solution.err
            m = solution.model
            cond = solution.conditions
            undec = solution.undec
            # Compute non-decision time
            if t_bin:
                t_param = getattr(self, 't{}'.format(cond['bin']))
            else:
                t_param = self.t
            shifts = int(t_param/m.dt) # truncate
            # Shift the distribution
            newcorr = np.zeros(corr.shape, dtype=corr.dtype)
            newerr = np.zeros(err.shape, dtype=err.dtype)
            if shifts > 0:
                newcorr[shifts:] = corr[:-shifts]
                newerr[shifts:] = err[:-shifts]
            elif shifts < 0:
                newcorr[:shifts] = corr[-shifts:]
                newerr[:shifts] = err[-shifts:]
            else:
                newcorr = corr
                newerr = err
            return Solution(newcorr, newerr, m, cond, undec)
    return NonDecisionTime

def make_model(model_settings):
    
    # model components:
    z = make_z(n_bins=model_settings['n_bins'], z_bin=model_settings['z_bin'])
    drift = make_drift(n_bins=model_settings['n_bins'], v_bin=model_settings['v_bin'], dc_bin=model_settings['dc_bin'])
    a = make_a(n_bins=model_settings['n_bins'], a_bin=model_settings['a_bin'], bound_collapse=model_settings['bound_collapse'])
    t = make_t(n_bins=model_settings['n_bins'], t_bin=model_settings['t_bin'])

    # limits:
    ranges = {
            'z':(0.05,0.95),    # starting point
            'v':(0,5),          # drift rate
            'b':(-5,5),         # drift bias
            'a':(0.1,5),        # bound
            'c':(0.01,100),     # hyperbolic collapse
            't':(0,2),          # non-decision time
            }

    # put together:
    model = Model(name='stimulus coding model / collapsing bound',
                IC=z(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in z.required_parameters}),
                drift=drift(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in drift.required_parameters}),
                bound=a(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in a.required_parameters}),
                overlay=OverlayChain(overlays=[t(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in t.required_parameters}),
                                                OverlayPoissonMixture(pmixturecoef=.05, rate=1)]), # OverlayUniformMixture(umixturecoef=.01)
                noise=NoiseConstant(noise=1),
                dx=.005, dt=.01, T_dur=model_settings['T_dur'])
    return model

def fit_per_subject(df, model):

    # sample:
    sample = Sample.from_pandas_dataframe(df=df, rt_column_name="rt", correct_column_name="response")

    # fit:
    fit_model = fit_adjust_model(sample=sample, model=model, lossfunction=LossBIC)

    return fit_model

def simulate_data(df, params, model_settings, subj_idx, nr_trials=10000):

    # make model
    model = make_model(model_settings=model_settings)
    
    # get param names:
    param_names = []
    for component in model.dependencies: 
        param_names = param_names + component.required_parameters
    
    # make list:
    params_to_set = []
    for p in param_names:
        if p[-1].isnumeric():
            ind = (params['subj_idx']==subj_idx)&(params['variable']==p[:-1])&(params['bin']==int(p[-1]))
        else:
            ind = (params['subj_idx']==subj_idx)&(params['variable']==p)
        params_to_set.append(Fitted(params.loc[ind, 'value']))

    # set:
    model.set_model_parameters(params_to_set)
    
    # compute number of trials to generate:
    trial_nrs = (df.loc[df['subj_idx']==subj_idx,:].groupby(['stimulus', 'bin']).count()['rt'] / 
                    sum(df.loc[df['subj_idx']==subj_idx,:].groupby(['stimulus', 'bin']).count()['rt']) * nr_trials).astype(int).reset_index()

    # generate:
    data_subj = []
    for stim in [-1,1]:
        for b in range(model_settings['n_bins']):
            if model_settings['n_bins'] > 1:
                samp = model.solve({'stimulus':stim, 'bin':b}).resample(int(trial_nrs.query('(stimulus=={})&(bin=={})'.format(stim, b))['rt']))
            else:
                samp = model.solve({'stimulus':stim}).resample(trial_nrs.query('(stimulus=={})'.format(stim))['rt'])
            data = pd.DataFrame({
                'rt': np.concatenate((samp.corr, samp.err)),
                'response': np.concatenate((np.ones(len(samp.corr)), np.zeros(len(samp.err))))
                })
            data['subj_idx'] = subj_idx
            data['stimulus'] = stim
            data['bin'] = b

            # remove RTs smaller than non-decision time:
            ind = (params['subj_idx']==subj_idx)&(params['variable']=='t')&(params['bin']==b)
            if sum(ind) == 0:
                ind = (params['subj_idx']==subj_idx)&(params['variable']=='t')
            data = data.loc[data['rt']>float(params.loc[ind, 'value']),:].reset_index()

            # append:
            data_subj.append(data)
    data_subj = pd.concat(data_subj).reset_index(drop=True)

    # add correct column:
    data_subj['correct'] = 0
    data_subj.loc[(data_subj['stimulus']==1)&(data_subj['response']==1), 'correct'] = 1
    data_subj.loc[(data_subj['stimulus']==-1)&(data_subj['response']==0), 'correct'] = 1
    
    return data_subj