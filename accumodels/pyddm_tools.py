#!/usr/bin/env python
# encoding: utf-8

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
# from ddm import set_N_cpus

from IPython import embed as shell

# set_N_cpus(16)

def get_param_names(sample, depends_on, param):

    if depends_on is None:
        unique_conditions = None
        names = [param]
    else:
        unique_conditions = [np.unique(np.concatenate((np.unique(sample.conditions[depends_on[i]][0]), 
                                        np.unique(sample.conditions[depends_on[i]][1])))) for i in range(len(depends_on))]
        if len(unique_conditions) == 1:
            names = ['{}{}'.format(param,i) for i in unique_conditions[0]]
        elif len(unique_conditions) == 2:
            names = ['{}{}.{}'.format(param,i,j) for i in unique_conditions[0] for j in unique_conditions[1]]

    return names, unique_conditions

def make_z(sample, z_depends_on=[None]):
    
    z_names, z_unique_conditions = get_param_names(sample=sample, depends_on=z_depends_on, param='z')

    class StartingPoint(InitialCondition):
        name = 'A starting point.'
        required_parameters = z_names
        if not z_depends_on is None:
            required_conditions = z_depends_on.copy()
        def get_IC(self, x, dx, conditions):
            pdf = np.zeros(len(x))
            if z_depends_on is None:
                z_param = self.z
            elif len(z_unique_conditions) == 1:
                z_param = getattr(self, 'z{}'.format(conditions[z_depends_on[0]]))
            elif len(z_unique_conditions) == 2:
                z_param = getattr(self, 'z{}.{}'.format(conditions[z_depends_on[0]],conditions[z_depends_on[1]]))
            pdf[int(len(pdf)*z_param)] = 1
            return pdf
    return StartingPoint

def make_drift(sample, v_depends_on=[None], b_depends_on=[None]):
    
    v_names, v_unique_conditions = get_param_names(sample=sample, depends_on=v_depends_on, param='v')
    b_names, b_unique_conditions = get_param_names(sample=sample, depends_on=b_depends_on, param='b')

    class DriftStimulusCoding(ddm.models.Drift):
        name = 'Drift'
        required_parameters = v_names + b_names
        required_conditions = ['stimulus']
        if (v_depends_on is not None):
            required_conditions = list(set(required_conditions+v_depends_on))
        if (b_depends_on is not None):
            required_conditions = list(set(required_conditions+b_depends_on))

        def get_drift(self, conditions, **kwargs):
            
            # v param:
            if v_depends_on is None:
                v_param = self.v
            elif len(v_unique_conditions) == 1:
                v_param = getattr(self, 'v{}'.format(conditions[v_depends_on[0]]))
            elif len(v_unique_conditions) == 2:
                v_param = getattr(self, 'v{}.{}'.format(conditions[v_depends_on[0]],conditions[v_depends_on[1]]))

            # b param:
            if b_depends_on is None:
                b_param = self.b
            elif len(b_unique_conditions) == 1:
                b_param = getattr(self, 'b{}'.format(conditions[b_depends_on[0]]))
            elif len(b_unique_conditions) == 2:
                b_param = getattr(self, 'b{}.{}'.format(conditions[b_depends_on[0]],conditions[b_depends_on[1]]))

            return b_param + (v_param * conditions['stimulus'])
    return DriftStimulusCoding

def make_a(sample, urgency, a_depends_on=[None], u_depends_on=[None]):
    
    a_names, a_unique_conditions = get_param_names(sample=sample, depends_on=a_depends_on, param='a')
    if urgency:
        u_names, u_unique_conditions = get_param_names(sample=sample, depends_on=u_depends_on, param='u') 
    else:
        u_names = []
    
    class BoundCollapsingHyperbolic(Bound):
        name = 'Hyperbolic collapsing bounds'
        required_parameters = a_names + u_names
        
        required_conditions = []
        if (a_depends_on is not None):
            required_conditions = list(set(required_conditions+a_depends_on))
        if (u_depends_on is not None):
            required_conditions = list(set(required_conditions+u_depends_on))
        
        def get_bound(self, t, conditions, **kwargs):
            
            # a param:
            if a_depends_on is None:
                a_param = self.a
            elif len(a_unique_conditions) == 1:
                a_param = getattr(self, 'a{}'.format(conditions[a_depends_on[0]]))
            elif len(a_unique_conditions) == 2:
                a_param = getattr(self, 'a{}.{}'.format(conditions[a_depends_on[0]],conditions[a_depends_on[1]]))
            
            if urgency:
                # u param:
                if u_depends_on is None:
                    u_param = self.u
                elif len(u_unique_conditions) == 1:
                    u_param = getattr(self, 'u{}'.format(conditions[u_depends_on[0]]))
                elif len(u_unique_conditions) == 2:
                    u_param = getattr(self, 'u{}.{}'.format(conditions[u_depends_on[0]],conditions[u_depends_on[1]]))
                
            # return:
                return a_param-(a_param*(t/(t+u_param)))
            else:
                return a_param
    return BoundCollapsingHyperbolic

def make_t(sample, t_depends_on=[None]):
    
    t_names, t_unique_conditions = get_param_names(sample=sample, depends_on=t_depends_on, param='t')
    
    class NonDecisionTime(Overlay):
        name = 'Non-decision time'
        required_parameters = t_names
        if not t_depends_on is None:
            required_conditions = t_depends_on.copy()
        def apply(self, solution):
            # Unpack solution object
            corr = solution.corr
            err = solution.err
            m = solution.model
            cond = solution.conditions
            undec = solution.undec
            
            # t param:
            if t_depends_on is None:
                t_param = self.t
            elif len(t_unique_conditions) == 1:
                t_param = getattr(self, 't{}'.format(cond[t_depends_on[0]]))
            elif len(t_unique_conditions) == 2:
                t_param = getattr(self, 't{}.{}'.format(cond[t_depends_on[0]],cond[t_depends_on[1]]))
            
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


# def make_z_reg():
#     class StartingPoint(InitialCondition):
#         name = 'A starting point.'
#         required_parameters = ['z', 'z_coeff']
#         required_conditions = ['physio']
#         def get_IC(self, x, dx, conditions):
#             pdf = np.zeros(len(x))
#             z_param = self.z + (self.z_coeff * conditions['physio'])
#             pdf[int(len(pdf)*z_param)] = 1
#             return pdf
#     return StartingPoint


# def make_drift_regression(v_reg=True, b_reg=True):
#     if v_reg:
#         v_names = ['v', 'v_coeff']
#     else:
#         v_names = ['v']
#     if b_bin:
#         b_names = ['b', 'b_coeff']
#     else:
#         b_names = ['b']
#     class DriftStimulusCoding(ddm.models.Drift):
#         name = 'Drift'
#         required_parameters = v_names + b_names
#         required_conditions = ['stimulus', 'physio']
#         def get_drift(self, conditions, **kwargs):
#             if v_reg and not b_reg:
#                 return self.b + (self.v * conditions['stimulus']) + (self.v_coeff * conditions['stimulus'] * conditions['physio'])
#             elif b_reg and not v_reg:
#                 return self.b + (self.v * conditions['stimulus']) + (self.b_coeff * conditions['physio'])
#             elif v_reg and b_reg:
#                 return self.b + (self.v * conditions['stimulus']) + (self.v_coeff * conditions['stimulus'] * conditions['physio']) + (self.b_coeff * conditions['physio'])
#     return DriftStimulusCoding

def make_model(sample, model_settings):
    
    # model components:
    z = make_z(sample=sample, 
                z_depends_on=model_settings['depends_on']['z'])
    drift = make_drift(sample=sample, 
                        v_depends_on=model_settings['depends_on']['v'], 
                        b_depends_on=model_settings['depends_on']['b'])
    a = make_a(sample=sample, 
                urgency=model_settings['urgency'], 
                a_depends_on=model_settings['depends_on']['a'], 
                u_depends_on=model_settings['depends_on']['u'])
    t = make_t(sample=sample, 
                t_depends_on=model_settings['depends_on']['t'])
    T_dur = model_settings['T_dur']

    # limits:
    ranges = {
            'z':(0.05,0.95),               # starting point
            'v':(0,5),                     # drift rate
            'b':(-5,5),                    # drift bias
            'a':(0.1,5),                   # bound
            # 'u':(-T_dur*10,T_dur*10),    # hyperbolic collapse
            'u':(0.01,T_dur*10),           # hyperbolic collapse
            't':(0,2),                     # non-decision time
            }

    # put together:
    model = Model(name='stimulus coding model / collapsing bound',
                IC=z(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in z.required_parameters}),
                drift=drift(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in drift.required_parameters}),
                bound=a(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in a.required_parameters}),
                overlay=OverlayChain(overlays=[t(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in t.required_parameters}),
                                                # OverlayUniformMixture(umixturecoef=.01)]),
                                                OverlayPoissonMixture(pmixturecoef=.01, rate=1)]),
                noise=NoiseConstant(noise=1),
                dx=.001, dt=.01, T_dur=T_dur)
    return model

def fit_model(df, model_settings):

    # sample:
    sample = Sample.from_pandas_dataframe(df=df, rt_column_name='rt', correct_column_name='response')

    # make model
    model = make_model(sample=sample, model_settings=model_settings)
    
    # fit:
    # fit_model = fit_adjust_model(sample=sample, model=model, lossfunction=LossBIC, fitparams={'maxiter':5000})
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
        try:
            params_to_set.append(Fitted(params.loc[ind, 'value']))
        except:
            shell()

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