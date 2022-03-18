#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from joblib import Parallel, delayed

import copy

import ddm
from ddm import Sample
from ddm import plot
from ddm import models
from ddm import Model, Fittable, Fitted, Bound, Overlay, Solution
from ddm.models.loss import LossFunction
from ddm.functions import fit_adjust_model, display_model
from ddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture, OverlayUniformMixture, InitialCondition, ICPoint, ICPointSourceCenter, LossBIC, LossRobustBIC, LossBICGonogo, LossRobustBICGonogo
# from ddm import set_N_cpus

from IPython import embed as shell

# set_N_cpus(16)

from model_components import OneAccumulatorNoise
from model_components import make_a
from model_components import make_drift

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

def make_model_one_accumulator(sample, model_settings):

    # parameters:
    ranges = {
            # 'v':(0.99,1.01),               # drift rate
            # 'b':(-0.21,-0.19),             # drift bias
            # 'k':(2.39,2.41),               # leak
            # 'z':(0.75,0.999),              # starting point --> translates into bound heigth 0-5
            'v':(0,10),                      # drift rate
            'b':(-5,5),                      # drift bias
            'k':(0,10),                      # leak
            'a':(0.5,1.5),                   # bound
            't':(0,0.1),                     # non-decision time
            'sz':(0,1),                      # starting point variability
            'lapse':(0.001,0.999),           # lapse rate
            'lapse_slope':(0,0.1),           # lapse rate slope
            'mixture':(0.001,0.999),         # mixture rate
            }

    bound = make_a(sample=sample, 
                a_depends_on=model_settings['depends_on']['a'])
    drift = make_drift(sample=sample, 
                a_depends_on=model_settings['depends_on']['a'],
                k_depends_on=model_settings['depends_on']['k'],
                v_depends_on=model_settings['depends_on']['v'])

    # shell()

    drift_params = {param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in drift.required_parameters}

    # if (model_settings['depends_on']['a'] is None):
    #     a0_value = Fittable(minval=ranges['a'][0], maxval=ranges['a'][1], default=1)
    #     a1_value = a0_value
    # else:
    #     a0_value = Fittable(minval=ranges['a'][0], maxval=ranges['a'][1], default=1)
    #     a1_value = Fittable(minval=ranges['a'][0], maxval=ranges['a'][1], default=1)
    
    # t_value = Fittable(minval=ranges['t'][0], maxval=ranges['t'][1], default=1)
    # v_value = Fittable(minval=ranges['v'][0], maxval=ranges['v'][1], default=1)
    # k_value = Fittable(minval=ranges['k'][0], maxval=ranges['k'][1], default=1)
    # b_value = 0

    # if model_settings['lapse']:
    #     lapse_value = Fittable(minval=ranges['lapse'][0], maxval=ranges['lapse'][1], default=0.5)
    # else:
    #     lapse_value = 0

    # if model_settings['lapse_slope']:
    #     lapse_slope_value = Fittable(minval=ranges['lapse_slope'][0], maxval=ranges['lapse_slope'][1], default=0.05)
    # else:
    #     lapse_slope_value = 0

    # if model_settings['non_dec_t']:
    #     tt_value = Fittable(minval=ranges['t'][0], maxval=ranges['t'][1], default=0.1)
    # else:
    #     tt_value = 0

    # if model_settings['sz']:
    #     sz_value = Fittable(minval=ranges['sz'][0], maxval=ranges['sz'][1], default=0.1)
    # else:
    #     sz_value = 0

    # if model_settings['mixture']:
    #     mixture_value = Fittable(minval=ranges['mixture'][0], maxval=ranges['mixture'][1], default=0.1)
    #     mu_value = Fittable(minval=0, maxval=0.5, default=0.2)
    #     sigma_value = Fittable(minval=0.001, maxval=0.5, default=0.05)
    # else:
    #     mixture_value = 0
    #     mu_value = 0
    #     sigma_value = 1

    # # components:
    # starting_point_components = {'a0':a0_value, 'a1':a1_value, 'sz':sz_value}
    # drift_components = {
    #                     'v':v_value,
    #                     'k':k_value,
    #                     'b':b_value,
    #                     'a0':a0_value,
    #                     'a1':a1_value,
    #                     'tt':tt_value,
    #                     }
    # mixture_components = {'mixture':mixture_value, 'mu':mu_value, 'sigma':sigma_value}
    # lapse_components = {'lapse':lapse_value, 'lapse_slope':lapse_slope_value}
    # non_dec_components = {'t':t_value, 't':t_value}


    # build model:
    from ddm.models import BoundConstant, OverlayChain
    model = Model(name='one accumulator model',
                IC=bound(**{key:value for (key, value) in drift_params.items() if 'a' in key}),
                drift=drift(**drift_params),
                bound=BoundConstant(B=10),
                # overlay=OverlayChain(overlays=[
                #                                 OneAccumulatorNonDecisionTime(**non_dec_components),
                #                                 OneAccumulatorOverlayGaussMixture(**mixture_components),
                #                                 OneAccumulatorOverlayEvidenceLapse(**lapse_components),
                #                                 ]),
                noise=OneAccumulatorNoise(noise=1),
                dx=model_settings['dx'], dt=model_settings['dt'], T_dur=model_settings['T_dur'])
    
    return model

def fit_model_one_accumulator(df, model_settings):

    from ddm import set_N_cpus
    set_N_cpus(14)

    # sample:
    sample = Sample.from_pandas_dataframe(df=df, rt_column_name='rt', correct_column_name='response')

    # make model
    model = make_model_one_accumulator(sample=sample, model_settings=model_settings)

    # return model

    # fit:
    # model = fit_adjust_model(sample=sample, model=model, lossfunction=LossBIC, fitparams={'maxiter':5000})
    # model = fit_adjust_model(sample=sample, model=model, lossfunction=LossRobustBIC)
    model = fit_adjust_model(sample=sample, model=model, lossfunction=LossRobustBICGonogo)
    
    # get params:
    # param_names = [component.required_parameters for component in model.dependencies]
    # param_names = [item for sublist in param_names for item in sublist]
    param_names = model.get_model_parameter_names()
    params = pd.DataFrame(np.atleast_2d([p.real for p in model.get_model_parameters()]), columns=param_names)
    params['bic'] = model.fitresult.value() 

    return params




def fit_model(df, model_settings):

    # sample:
    sample = Sample.from_pandas_dataframe(df=df, rt_column_name='rt', correct_column_name='response')

    # make model
    model = make_model(sample=sample, model_settings=model_settings)

    # fit:
    # model = fit_adjust_model(sample=sample, model=model, lossfunction=LossBIC, fitparams={'maxiter':5000})
    model = fit_adjust_model(sample=sample, model=model, lossfunction=LossRobustBIC)

    # get params:
    # param_names = [component.required_parameters for component in model.dependencies]
    # param_names = [item for sublist in param_names for item in sublist]
    param_names = model.get_model_parameter_names()
    params = pd.DataFrame(np.atleast_2d([p.real for p in model.get_model_parameters()]), columns=param_names)
    params['bic'] = model.fitresult.value()

    return params

def simulate_data(df, params, model_settings, subj_idx, nr_trials=10000):

    # sample:
    if np.mean(np.isnan(df.loc[df['response']==0,'rt']))==1:
        df['rt'] = np.random.rand(df.shape[0])
        gonogo = True
        print('gonogo data!')
    else:
        gonogo = False
    sample = Sample.from_pandas_dataframe(df=df, rt_column_name='rt', correct_column_name='response')

    # make model:
    model = make_model(sample=sample, model_settings=model_settings)

    # remove laps rate:
    params['umixturecoef'] = 0.05
    params['pmixturecoef'] = 0.05
    params['rate'] = 1
    params['noise'] = 1

    # set fitted parameters:
    # param_names = [component.required_parameters for component in model.dependencies]
    # param_names = [item for sublist in param_names for item in sublist]
    param_names = model.get_model_parameter_names()
    print(param_names)
    params_to_set = [Fitted(params.loc[(params['subj_idx']==subj_idx), param]) for param in param_names]
    model.set_model_parameters(params_to_set)
    
    # compute number of trials to generate:
    df.loc[:,'trials'] = 1
    trial_nrs = df.groupby(model.required_conditions).count()['trials']
    trial_nrs = (trial_nrs / sum(trial_nrs) * nr_trials).reset_index()
    trial_nrs['trials'] = trial_nrs['trials'].astype(int)
    
    # generate:
    data_subj = []
    for ids, d in trial_nrs.groupby(model.required_conditions):
        
        if isinstance(ids, int):
            ids = [ids]
        
        if len(ids) == 3:
            t = 't{}'.format(ids[1], ids[2])
        elif len(ids) == 2:
            t = 't{}'.format(ids[1])
        else:
            t = 't'

        # generate data:
        samp = model.solve({key:value for value, key in zip(ids, model.required_conditions)}).resample(int(d['trials']))

        # to dataframe:
        data = pd.DataFrame({
            'rt': np.concatenate((samp.corr, samp.err)),
            'response': np.concatenate((np.ones(len(samp.corr)), np.zeros(len(samp.err))))
            })
        data['subj_idx'] = subj_idx
        for value, key in zip(ids, model.required_conditions):
            data[key] = value
        
        # remove RTs smaller than non-decision time:
        # non_decision_time = params.loc[(params['subj_idx']==subj_idx), t]
        non_decision_time = 0
        data = data.loc[data['rt']>non_decision_time,:].reset_index()

        # append:
        data_subj.append(data)
    data_subj = pd.concat(data_subj).reset_index(drop=True)

    # add correct column:
    data_subj['correct'] = 0
    data_subj.loc[(data_subj['stimulus']==1)&(data_subj['response']==1), 'correct'] = 1
    data_subj.loc[(data_subj['stimulus']==-1)&(data_subj['response']==0), 'correct'] = 1
    
    # gonogo data?
    if gonogo:
        data_subj.loc[data_subj['response']==0, 'rt'] = np.nan

    return data_subj

def sample_from_condition(model, ids, nr_trials):

    if isinstance(ids, int):
        ids = [ids]

    # sample:
    samp = model.solve({key:value for value, key in zip(ids, model.required_conditions)}).resample(nr_trials)

    # to dataframe:
    df = pd.DataFrame({'rt': samp.corr,})
    df['response'] = (df['rt']>0).astype(int)
    df.loc[df['rt']<=0, 'rt'] = np.NaN

    # meta:
    for value, key in zip(ids, model.required_conditions):
        df[key] = value
    
    return df

def simulate_data_gonogo(df, params, model_settings, nr_trials=10000, rt_cutoff=0.1, n_jobs=24):

    # previous choice:
    choice_p_fraction = df['choice_p'].mean()

    # generate trials:
    noise_dur_mean = 5
    noise_dur_max = 11
    sig_dur = 3
    # tmax = noise_dur_max + sig_dur
    dfs = []
    for reward in [0,1]:
        noise_durs = np.random.exponential(noise_dur_mean, nr_trials)
        noise_durs[noise_durs>=noise_dur_max] = noise_dur_max
        # noise_durs = noise_durs[noise_durs<=noise_dur_max]
        d = pd.DataFrame({'reward': np.repeat(reward, len(noise_durs)), 'noise_dur': noise_durs, 'duration': np.repeat(sig_dur, len(noise_durs))})
        dfs.append(d)
    df = pd.concat(dfs)
    df['response'] = 1
    df['rt'] = 1

    # bin
    shift = 'min'
    df['start'] = df['noise_dur'].copy()
    df.loc[(df['start']<11), 'bins'] = pd.cut(df.loc[(df['start']<11), 'start'], bins=[-1,1,2,3,4,5,6,7,8,9,10,11], labels=False)
    df.loc[df['start']==11, 'bins'] = df['bins'].max()+1
    print(df.groupby(['bins'])['start'].mean())
    if shift == 'min':
        df['start'] = df.groupby(['bins'])['start'].transform(lambda x: x.min())
    if shift == 'max':
        df['start'] = df.groupby(['bins'])['start'].transform(lambda x: x.max())
    if shift == 'mean':
        df['start'] = df.groupby(['bins'])['start'].transform(lambda x: x.mean())
    print(df.groupby(['bins'])['start'].mean())
    df['start'] = df['start'].round(1)

    # add choice_p:
    df['choice_p'] = (np.random.rand(df.shape[0])<choice_p_fraction).astype(int)

    # make sample:
    sample = Sample.from_pandas_dataframe(df=df, rt_column_name='rt', correct_column_name='response')

    # make model:
    model = make_model_one_accumulator(model_settings=model_settings)

    # set fitted parameters:
    param_names = model.get_model_parameter_names()
    try:
        params_to_set = [Fitted(params[param]) for param in param_names]
    except:
        print(param_names)
    model.set_model_parameters(params_to_set)
    
    # compute number of trials to generate:
    nr_trials = df.shape[0]
    df.loc[:,'trials'] = 1
    trial_nrs = df.groupby(model.required_conditions).count()['trials']
    trial_nrs = (trial_nrs / sum(trial_nrs) * nr_trials).reset_index()
    trial_nrs['trials'] = trial_nrs['trials'].astype(int)
    print(trial_nrs)
    
    # resample:
    res = Parallel(n_jobs=n_jobs)(delayed(sample_from_condition)(model=model, ids=ids, nr_trials=int(d['trials']))
                                    for ids, d in trial_nrs.groupby(model.required_conditions))
    
    # concat:
    df_sim = pd.concat(res)

    return df_sim

    # df_sim.loc[ (df_sim['hit']==1) & ((df_sim['rt']-df_sim['start'])<0.01),:]