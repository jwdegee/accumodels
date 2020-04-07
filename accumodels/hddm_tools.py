#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import scipy as sp
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import hddm
import kabuki
from joblib import Memory
from joblib import Parallel, delayed
from IPython import embed as shell

def fit_ddm_per_group(data, model, model_dir, model_name, samples=5000, burn=1000, thin=1, n_models=3, n_jobs=12):
    
    res = Parallel(n_jobs=n_jobs)(delayed(fit_ddm_hierarchical)(data, model, model_dir, model_name, samples, burn, thin, model_id) for model_id in range(n_models))

def fit_ddm_hierarchical(df, model_settings, model_dir, model_name, samples=5000, burn=1000, thin=1, model_id=0):
    
    # fix depends_on:
    depends_on = model_settings['depends_on']
    if depends_on is not None:
        if 'dc' in depends_on:
            depends_on['dc'] = depends_on['b']
        depends_on.pop('b', None)
        depends_on.pop('u', None)

    # fit:
    m = hddm.HDDMStimCoding(df, stim_col='stimulus', split_param='v', drift_criterion=True, bias=True, 
                            depends_on=depends_on, include=('sv'), group_only_nodes=['sv'], p_outlier=0)
    m.find_starting_values()
    m.sample(samples, burn=burn, thin=thin, dbname=os.path.join(model_dir, '{}_{}.db'.format(model_name, model_id)), db='pickle')
    m.save(os.path.join(model_dir, '{}_{}.hddm'.format(model_name, model_id)))
    
    # params:    
    params = m.gen_stats()['mean'].reset_index()
    params.columns = ['variable', 'value']
    params = params.loc[['subj' in p for p in params['variable']],:]
    params['subj_idx'] = [p.split('.')[-1] for p in params['variable']]
    params['subj_idx'] = params['subj_idx'].astype(int)
    params['variable'] = [p.replace('_subj', '') for p in params['variable']]
    params['variable'] = [p.replace('(', '') for p in params['variable']]
    params['variable'] = [p.replace(')', '') for p in params['variable']]
    params['variable'] = [p.split('.')[0] for p in params['variable']]
    params = params.pivot(index='subj_idx', columns='variable')
    params.columns = params.columns.droplevel(level=0)
    params.columns.name = None
    params = params.reset_index()
    params = params.sort_values(by=['subj_idx'])
    
    # fix columns:
    params.columns = [p if not 'dc' else p.replace('dc', 'b') for p in params.columns]

    # fix values:
    params.loc[:, [p[0]=='a' for p in params.columns]] = params.loc[:, [p[0]=='a' for p in params.columns]] / 2 
    params['noise'] = 1
    params['umixturecoef'] = 0

    return params

def load_ddm_per_group(model_dir, model_name, n_models=3):
    
    models = [kabuki.utils.load(os.path.join(model_dir, '{}_{}.hddm'.format(model_name, model_id))) for model_id in range(n_models)] 
    
    return models

def fit_ddm_per_subject(data, analysis_info, model, model_dir, model_name, n_runs=5, n_jobs=12):

    res = Parallel(n_jobs=n_jobs)(delayed(fit_ddm_subject)(subj_data, analysis_info, subj_idx, model, model_dir, model_name, n_runs) for subj_idx, subj_data in data.groupby('subj_idx'))
    
    # # serial:
    # res = []
    # for subj_idx, subj_data in df.groupby('subj_idx'):
    #     res.append( fit_ddm_subject(subj_data, subj_idx, model, model_dir, model_name, n_runs) )
    
    res = pd.concat(res, axis=0)
    res.to_csv(os.path.join(model_dir, '{}_params_flat.csv'.format(model_name)))
    return res
    
def fit_ddm_subject(data, analysis_info, subj_idx, model, model_dir, model_name, n_runs=5):
    
    import hddm
    import pandas as pd
    data = data.loc[data["subj_idx"]==subj_idx,:]
    exec('m = {}'.format(model), locals(), globals())

    # starting values:
    if sum(np.isnan(data.loc[data['response']==0, 'rt'])) == sum(data['response']==0):
        # pass
        print('finding starting values via basic model!')
        basic_m = hddm.HDDMStimCoding(data, stim_col='stimulus', split_param='v', drift_criterion=False, bias=False, p_outlier=0)
        basic_m.approximate_map()
        basic_m.optimize('gsquare', quantiles=analysis_info['quantiles'], n_runs=n_runs)
        values_m = m.values
        
        # for v in ['v', 'a', 't', 'z', 'z_trans', 'dc']:
        names = []
        best_values = []
        for v in ['v', 'a', 't',]:
            values_to_set = []
            for p in m.values:
                if '{}'.format(v) == p:
                    values_to_set.append(p)
                elif '{}('.format(v) in p:
                    values_to_set.append(p)
            for p in values_to_set:
                try:
                    values_m[p] = basic_m.values['{}'.format(v)]
                    names.append(p)
                    best_values.append( basic_m.values['{}'.format(v)])
                except:
                    pass
        m.set_values(dict(list(zip(names, best_values))))
        print(m.values)
    else:
        print('finding starting values!')
        m.approximate_map()
        print(m.values)
    
    # optimize:
    print('fitting model!')
    m.optimize('gsquare', quantiles=analysis_info['quantiles'], n_runs=n_runs)
    res = pd.concat((pd.DataFrame([m.values], index=[subj_idx]), pd.DataFrame([m.bic_info], index=[subj_idx])), axis=1)
    # res['aic'] = m.aic
    # res['bic'] = m.bic

    return res

def load_ddm_per_subject(model_dir, model_name):
    
    return pd.read_csv(os.path.join(model_dir, '{}_params_flat.csv'.format(model_name))).drop('Unnamed: 0', 1)

def get_point_estimates_hierarchical(results, parameters):

    results = results.loc[[('_subj' in i) for i in results.index], :]

    all_params = []
    for p in parameters:
        temp = results.loc[[('{}_subj'.format(p) in i) for i in results.index], :]
        if sum(['(' in i for i in temp.index]) > 0:
            conditions = np.unique([temp.index[i][temp.index[i].find('(')+1:temp.index[i].find(')')] for i in range(temp.shape[0])])
        else:
            conditions = None
        params = []
        if conditions != None:
            for c in conditions:
                ind = np.array([temp.index[i][temp.index[i].find('(')+1:temp.index[i].find(')')] == c for i in range(temp.shape[0])])
                values = temp.loc[ind, 'mean'].reset_index()
                values['condition'] = c
                values.columns = ['subj_idx', p, 'condition']
                values['subj_idx'] = np.array([temp.index[i].split('.')[-1] for i in range(values.shape[0])], dtype=int)
                params.append(values)
        else:
            values = temp.loc[:, 'mean'].reset_index()
            values.columns = ['subj_idx', p]
            values['condition'] = 0
            values['subj_idx'] = np.arange(values.shape[0])
            params.append(values)
        params = pd.concat(params, axis=0).reset_index(drop=True)
        all_params.append(params)
    
    df = pd.concat(all_params, axis=1)
    df = df.loc[:,~df.columns.duplicated()]
    df['subj_idx'] = df['subj_idx'].astype(int)
    # df['condition'] = df['condition'].astype(int)
    for p in parameters:
        df[p] = df[p].astype(float)

    return df

def get_point_estimates_flat(results):
    
    columns = [c for c in results.columns if not (c == 'z_trans')|(c == 'bic')|(c == 'likelihood')|(c == 'penalty')]
    params = results.loc[:,columns]

    a = []
    for c in params.columns:
        if not '(' in c:
            param = c
            values = params[c]
            a.append(pd.DataFrame({'param':param, 'values':values, 'cond0':1, 'cond1':0, 'cond2':0, 'subj_idx':np.arange(len(values))}))
        elif ('(' in c) and not ('.' in c):
            param = c.split('(')[0]
            values = params[c]
            a.append(pd.DataFrame({'param':param, 'values':values, 'cond0':0, 'cond1':int(c.split('(')[1][:-1])+1, 'cond2':0, 'subj_idx':np.arange(len(values))}))
        elif ('(' in c) and ('.' in c):
            param = c.split('(')[0]
            values = params[c]
            a.append(pd.DataFrame({'param':param, 'values':values, 'cond0':0, 'cond1':int(c.split('(')[1][:-1].split('.')[0])+1, 'cond2':int(c.split('(')[1][:-1].split('.')[1])+1, 'subj_idx':np.arange(len(values))}))
    a = pd.concat(a, axis=0)
    
    a['condition'] = np.NaN
    unique = a.loc[:,['cond0', 'cond1', 'cond2']].drop_duplicates()
    for i in range(unique.shape[0]):
        a.loc[(a['cond0']==unique.iloc[i]['cond0'])&(a['cond1']==unique.iloc[i]['cond1'])&(a['cond2']==unique.iloc[i]['cond2']), 'condition'] = i
    a['condition'] = a['condition'].astype(int)
        
    return a
    
def plot_posteriors(traces, comparisons, conditions=['low', 'high'], colors=['blue', 'red']):
    
    # stats
    p_values = []
    for p in comparisons:
        d = [traces[p[0]], traces[p[2]]]
        stat = np.mean(d[0] > d[1])
        p_values.append(min(stat, 1-stat))
    p_values = np.array(p_values)

    # plot:
    fig, axes = plt.subplots(nrows=1, ncols=len(comparisons), figsize=(len(comparisons)*1.5,2.5))
    ax_nr = 0
    for i, p in enumerate(comparisons):
        d = [traces[p[t]] for t in range(len(conditions))]
        ax = axes[ax_nr]
        for d, label, c in zip(d, conditions, colors):
            sns.kdeplot(d, vertical=True, shade=True, color=c, label=label, ax=ax)
        ax.set_xlabel('Posterior probability')
        ax.set_title(str(p)+'\np={}'.format(round(p_values[i],4)))
        ax.set_xlim(xmin=0)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)
            ax.tick_params(width=0.5)
        ax_nr+=1
    sns.despine(offset=10, trim=True)
    axes[0].set_ylabel('Parameter estimate (a.u.)')
    plt.tight_layout()
    return fig

def conditional_response_plot(df_group, df_sim_group=None, quantiles=[0, 0.1, 0.3, 0.5, 0.7, 0.9,], bin_nr_on_x=False, xlim=None, ylim=None, color='orange', ax=None):
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # condition response plots:
    df_group.loc[:,'rt_bin'] = df_group.groupby(['subj_idx'])['rt'].apply(pd.qcut, quantiles, labels=False)
    d = df_group.groupby(['subj_idx', 'rt_bin']).mean().reset_index()
    if bin_nr_on_x:
        ax.errorbar(x=np.unique(d['rt_bin']), y=d.groupby(['rt_bin'])["response"].mean(), 
                xerr=d.groupby(['rt_bin'])["rt"].sem(), yerr=d.groupby(['rt_bin'])["response"].sem(), 
                fmt='-o', color=color, markersize=6)
    else:
        ax.errorbar(x=d.groupby(['rt_bin'])["rt"].mean(), y=d.groupby(['rt_bin'])["response"].mean(), 
                xerr=d.groupby(['rt_bin'])["rt"].sem(), yerr=d.groupby(['rt_bin'])["response"].sem(), 
                fmt='-o', color=color, markersize=6)
    if df_sim_group is not None:
        df_sim_group.loc[:,'rt_bin'] = df_sim_group.groupby(['subj_idx'])['rt'].apply(pd.qcut, quantiles, labels=False)
        d = df_sim_group.groupby(['subj_idx', 'rt_bin']).mean().reset_index()
        if bin_nr_on_x:
            ax.errorbar(x=np.unique(d['rt_bin']), y=d.groupby(['rt_bin'])["response"].mean(), 
            # xerr=d.groupby(['rt_bin'])["rt"].sem(), yerr=d.groupby(['rt_bin'])["response"].sem(), 
            fmt='-', color='black', markersize=6)
        else:
            ax.errorbar(x=d.groupby(['rt_bin'])["rt"].mean(), y=d.groupby(['rt_bin'])["response"].mean(), 
            # xerr=d.groupby(['rt_bin'])["rt"].sem(), yerr=d.groupby(['rt_bin'])["response"].sem(), 
            fmt='-', color='black', markersize=6)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title('Conditional response')
    ax.set_xlabel('RT (quantiles)')
    ax.set_ylabel('P(bias)')

    res_emp = df_group.groupby(['subj_idx', 'rt_bin'])["rt", "response"].mean().reset_index()
    res_emp['data'] = 'emp'
    res_sim = df_sim_group.groupby(['subj_idx', 'rt_bin'])["rt", "response"].mean().reset_index()
    res_sim['data'] = 'sim'
    residuals = pd.concat((res_emp, res_sim), axis=0)
    
    # if bin_nr_on_x:
    # else:
    #     res_resp = abs((df_group.groupby(['subj_idx', 'rt_bin'])["response"].mean().groupby(['subj_idx']).mean() - df_sim_group.groupby(['subj_idx', 'rt_bin'])["response"].mean().groupby(['subj_idx']).mean()))
    #     res_rt = abs((df_group.groupby(['subj_idx', 'rt_bin'])["rt"].mean().groupby(['subj_idx']).mean() - df_sim_group.groupby(['subj_idx', 'rt_bin'])["rt"].mean().groupby(['subj_idx']).mean()))
    #     residuals = np.sqrt((res_resp**2) + (res_rt**2))
    return ax, residuals

def get_choice(row):
    
    if row.condition == 'present':
        if row.response == 1:
            return 1
        else:
            return 0
    elif row.condition == 'absent':
        if row.response == 0:
            return 1
        else:
            return 0

def simulate_data(a, v, t, z, dc, sv=0, sz=0, st=0, condition=0, nr_trials1=1000, nr_trials2=1000):
    
    """
    Simulates stim-coded data.
    """
    
    parameters1 = {'a':a, 'v':v+dc, 't':t, 'z':z, 'sv':sv, 'sz': sz, 'st': st}
    parameters2 = {'a':a, 'v':v-dc, 't':t, 'z':1-z, 'sv':sv, 'sz': sz, 'st': st}
    df_sim1, params_sim1 = hddm.generate.gen_rand_data(params=parameters1, size=nr_trials1, subjs=1, subj_noise=0)
    df_sim1['condition'] = 'present'
    df_sim2, params_sim2 = hddm.generate.gen_rand_data(params=parameters2, size=nr_trials2, subjs=1, subj_noise=0)
    df_sim2['condition'] = 'absent'
    df_sim = pd.concat((df_sim1, df_sim2))
    df_sim['bias_response'] = df_sim.apply(get_choice, 1)
    df_sim['correct'] = df_sim['response'].astype(int)
    df_sim['response'] = df_sim['bias_response'].astype(int)
    df_sim['stimulus'] = np.array((np.array(df_sim['response']==1) & 
                                    np.array(df_sim['correct']==1)) + (np.array(df_sim['response']==0) & 
                                    np.array(df_sim['correct']==0)), dtype=int)
    df_sim['condition'] = condition
    df_sim = df_sim.drop(columns=['bias_response'])
    
    return df_sim

def summary_plot_group(df_group, df_sim_group=None, quantiles=[0, 0.1, 0.3, 0.5, 0.7, 0.9,], xlim=None):

    # # remove NaNs:
    # df = df.loc[~pd.isna(df.rt),:]
    # if df_sim is not None:
    #     df_sim = df_sim.loc[~pd.isna(df_sim.rt),:]

    # step_size = 0.125
    step_size = 0.15

    a = np.percentile(df_group.loc[~np.isnan(df_group['rt']), 'rt'], 99)+0.25
    max_rt_group = np.ceil(a * 2)/2
    if max_rt_group <= 1.5:
        x_grid = np.arange(-max_rt_group-step_size, max_rt_group+step_size, step_size)
    else:
        x_grid = np.arange(-max_rt_group-step_size, max_rt_group+step_size, step_size)

    nr_subjects = len(np.unique(df_group['subj_idx']))

    counts_acc_emp = []
    counts_acc_sim = []
    counts_resp_emp = []
    counts_resp_sim = []
    for s in np.unique(df_group['subj_idx']):
        df = df_group.copy().loc[(df_group['subj_idx']==s),:]
        df_sim = df_sim_group.copy().loc[(df_sim_group['subj_idx']==s),:]
        df['rt_acc'] = df['rt'].copy()
        df.loc[df['correct']==0, 'rt_acc'] = df.loc[df['correct']==0, 'rt_acc'] * -1
        df['rt_resp'] = df['rt'].copy()
        df.loc[df['response']==0, 'rt_resp'] = df.loc[df['response']==0, 'rt_resp'] * -1
        df_sim['rt_acc'] = df_sim['rt'].copy()
        df_sim.loc[df_sim['correct']==0, 'rt_acc'] = df_sim.loc[df_sim['correct']==0, 'rt_acc'] * -1
        df_sim['rt_resp'] = df_sim['rt'].copy()
        df_sim.loc[df_sim['response']==0, 'rt_resp'] = df_sim.loc[df_sim['response']==0, 'rt_resp'] * -1
        max_rt = np.percentile(df.loc[~np.isnan(df['rt']), 'rt'], 95) + 0.1
        bins = np.linspace(-max_rt,max_rt,21)
        
        # rt distributions correct vs error:
        N, bins = np.histogram(df.loc[~np.isnan(df['rt_acc']), 'rt_acc'], bins=x_grid, density=True)
        counts_acc_emp.append(N)  
        N, bins = np.histogram(df_sim.loc[~np.isnan(df_sim['rt_acc']), 'rt_acc'], bins=x_grid, density=True)
        counts_acc_sim.append(N)
        N, bins = np.histogram(df.loc[~np.isnan(df['rt_resp']), 'rt_resp'], bins=x_grid, density=True)
        counts_resp_emp.append(N)  
        N, bins = np.histogram(df_sim.loc[~np.isnan(df_sim['rt_resp']), 'rt_resp'], bins=x_grid, density=True)
        counts_resp_sim.append(N)
    counts_acc_emp = np.vstack(counts_acc_emp)
    counts_acc_sim = np.vstack(counts_acc_sim)
    counts_resp_emp = np.vstack(counts_resp_emp)
    counts_resp_sim = np.vstack(counts_resp_sim)
    
    fig = plt.figure(figsize=(8,2))

    # histogram:
    ax = fig.add_subplot(1,4,1)
    x = x_grid[1:]
    ax.fill_between(x[x<=0], counts_acc_emp.mean(axis=0)[x<=0], color='red', step="pre", alpha=0.4)
    ax.fill_between(x[x>0], counts_acc_emp.mean(axis=0)[x>0], color='green', step="pre", alpha=0.4)
    ax.step(x, counts_acc_sim.mean(axis=0), color='k')
    ax.set_title('P(correct)={} +-{}'.format(
                                round(df_group.groupby('subj_idx')['correct'].mean().mean(), 3),
                                round(df_group.groupby('subj_idx')['correct'].mean().std(), 3),))
    ax.set_xlabel('RT (s)')
    ax.set_ylabel('Trials (prob. dens.)')

    # condition accuracy plots:
    ax = fig.add_subplot(1,4,2)
    df_group.loc[:,'rt_bin'] = df_group.groupby(['subj_idx'])['rt'].apply(pd.qcut, quantiles, labels=False)
    d = df_group.groupby(['subj_idx', 'rt_bin']).mean().reset_index()
    ax.errorbar(x=d.groupby(['rt_bin'])["rt"].mean(), y=d.groupby(['rt_bin'])["correct"].mean(), 
                xerr=d.groupby(['rt_bin'])["rt"].sem(), yerr=d.groupby(['rt_bin'])["correct"].sem(), 
                fmt='-o', color='orange', markersize=10)
    if df_sim_group is not None:
        df_sim_group.loc[:,'rt_bin'] = df_sim_group.groupby(['subj_idx'])['rt'].apply(pd.qcut, quantiles, labels=False)
        d = df_sim_group.groupby(['subj_idx', 'rt_bin']).mean().reset_index()
        ax.errorbar(x=d.groupby(['rt_bin'])["rt"].mean(), y=d.groupby(['rt_bin'])["correct"].mean(), 
            xerr=d.groupby(['rt_bin'])["rt"].sem(), yerr=d.groupby(['rt_bin'])["correct"].sem(), 
            fmt='-o', color='black', markersize=5)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_ylim(0, 1)
    ax.set_title('Conditional accuracy')
    ax.set_xlabel('RT (quantiles)')
    ax.set_ylabel('P(correct)')

    # histogram:
    ax = fig.add_subplot(1,4,3)
    x = x_grid[1:]
    ax.fill_between(x[x<=0], counts_resp_emp.mean(axis=0)[x<=0], color='cyan', step="pre", alpha=0.4)
    ax.fill_between(x[x>0], counts_resp_emp.mean(axis=0)[x>0], color='magenta', step="pre", alpha=0.4)
    ax.step(x, counts_resp_sim.mean(axis=0), color='k')
    
    if np.isnan(df_group['rt']).sum() > 0:
        bar_width = 1.5
        fraction_yes = df_group.groupby(['subj_idx'])['response'].mean().mean()
        fraction_yes_sim = df_sim_group.groupby(['subj_idx'])['response'].mean().mean()
        no_height = (1-fraction_yes)/fraction_yes/bar_width
        no_height_sim = (1-fraction_yes_sim)/fraction_yes_sim/bar_width

        ax.bar(x=-1.5, height=no_height, width=bar_width, alpha=0.5, color='cyan', align='center')
        ax.hlines(y=no_height_sim, xmin=-1.5-(bar_width/2), xmax=-1.5+(bar_width/2), lw=1, colors='black',)
        ax.vlines(x=-1.5-(bar_width/2), ymin=0, ymax=no_height_sim, lw=1, colors='black')
        ax.vlines(x=-1.5+(bar_width/2), ymin=0, ymax=no_height_sim, lw=1, colors='black')

    ax.set_title('P(bias)={} +-{}'.format(
                                round(df_group.groupby('subj_idx')['response'].mean().mean(), 3),
                                round(df_group.groupby('subj_idx')['response'].mean().std(), 3),))
    ax.set_xlabel('RT (s)')
    ax.set_ylabel('Trials (prob. dens.)')

    # condition response plots:
    ax = fig.add_subplot(1,4,4)
    df_group.loc[:,'rt_bin'] = df_group.groupby(['subj_idx'])['rt'].apply(pd.qcut, quantiles, labels=False)
    d = df_group.groupby(['subj_idx', 'rt_bin']).mean().reset_index()
    ax.errorbar(x=d.groupby(['rt_bin'])["rt"].mean(), y=d.groupby(['rt_bin'])["response"].mean(), 
                xerr=d.groupby(['rt_bin'])["rt"].sem(), yerr=d.groupby(['rt_bin'])["response"].sem(), 
                fmt='-o', color='orange', markersize=10)
    if df_sim is not None:
        df_sim_group.loc[:,'rt_bin'] = df_sim_group.groupby(['subj_idx'])['rt'].apply(pd.qcut, quantiles, labels=False)
        d = df_sim_group.groupby(['subj_idx', 'rt_bin']).mean().reset_index()
        ax.errorbar(x=d.groupby(['rt_bin'])["rt"].mean(), y=d.groupby(['rt_bin'])["response"].mean(), 
            xerr=d.groupby(['rt_bin'])["rt"].sem(), yerr=d.groupby(['rt_bin'])["response"].sem(), 
            fmt='-o', color='black', markersize=5)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_ylim(0, 1)
    ax.set_title('Conditional response')
    ax.set_xlabel('RT (quantiles)')
    ax.set_ylabel('P(bias)')

    sns.despine(offset=3, trim=True)
    plt.tight_layout()

    return fig

def summary_plot(df_group, df_sim_group=None, quantiles=[0, 0.1, 0.3, 0.5, 0.7, 0.9,], xlim=None):

    # # remove NaNs:
    # df = df.loc[~pd.isna(df.rt),:]
    # if df_sim is not None:
    #     df_sim = df_sim.loc[~pd.isna(df_sim.rt),:]

    nr_subjects = len(np.unique(df_group['subj_idx']))

    fig = plt.figure(figsize=(10,nr_subjects*2))
    plt_nr = 1
    for s in np.unique(df_group['subj_idx']):
        
        print(s)

        df = df_group.copy().loc[(df_group['subj_idx']==s),:]
        df_sim = df_sim_group.copy().loc[(df_sim_group['subj_idx']==s),:]
        df['rt_acc'] = df['rt'].copy()
        df.loc[df['correct']==0, 'rt_acc'] = df.loc[df['correct']==0, 'rt_acc'] * -1
        df['rt_resp'] = df['rt'].copy()
        df.loc[df['response']==0, 'rt_resp'] = df.loc[df['response']==0, 'rt_resp'] * -1
        df_sim['rt_acc'] = df_sim['rt'].copy()
        df_sim.loc[df_sim['correct']==0, 'rt_acc'] = df_sim.loc[df_sim['correct']==0, 'rt_acc'] * -1
        df_sim['rt_resp'] = df_sim['rt'].copy()
        df_sim.loc[df_sim['response']==0, 'rt_resp'] = df_sim.loc[df_sim['response']==0, 'rt_resp'] * -1
        max_rt = np.percentile(df.loc[~np.isnan(df['rt']), 'rt'], 95) + 0.1
        bins = np.linspace(-max_rt,max_rt,21)
        
        # rt distributions correct vs error:
        ax = fig.add_subplot(nr_subjects,4,plt_nr)
        N, bins, patches = ax.hist(df.loc[:, 'rt_acc'], bins=bins, 
                                   density=True, color='green', alpha=0.5)       
        for bin_size, bin, patch in zip(N, bins, patches):
            if bin < 0:
                plt.setp(patch, 'facecolor', 'r')
        if df_sim is not None:
            ax.hist(df_sim.loc[:, 'rt_acc'], bins=bins, density=True, 
                    histtype='step', color='k', alpha=1, label=None)   
        ax.set_title('P(correct)={}'.format(round(df.loc[:, 'correct'].mean(), 3),))
        ax.set_xlabel('RT (s)')
        ax.set_ylabel('Trials (prob. dens.)')
        plt_nr += 1

        # condition accuracy plots:
        ax = fig.add_subplot(nr_subjects,4,plt_nr)
        df.loc[:,'rt_bin'] = pd.qcut(df['rt'], quantiles, labels=False)
        d = df.groupby(['rt_bin']).mean().reset_index()
        ax.errorbar(d.loc[:, "rt"], d.loc[:, "correct"], fmt='-o', color='orange', markersize=10)
        if df_sim is not None:
            df_sim.loc[:,'rt_bin'] = pd.qcut(df_sim['rt'], quantiles, labels=False)
            d = df_sim.groupby(['rt_bin']).mean().reset_index()
            ax.errorbar(d.loc[:, "rt"], d.loc[:, "correct"], fmt='x', color='k', markersize=6)
        if xlim:
            ax.set_xlim(xlim)
        ax.set_ylim(0, 1)
        ax.set_title('Conditional accuracy')
        ax.set_xlabel('RT (quantiles)')
        ax.set_ylabel('P(correct)')
        plt_nr += 1
        
        # rt distributions response 1 vs 0:
        ax = fig.add_subplot(nr_subjects,4,plt_nr)
        if np.isnan(df['rt']).sum() > 0:
            bar_width = 1
            fraction_yes = df['response'].mean()
            fraction_yes_sim = df_sim['response'].mean()
            hist, edges = np.histogram(df.loc[:, 'rt_resp'], bins=bins, density=True,)
            hist = hist * fraction_yes
            hist_sim, edges_sim = np.histogram(df_sim.loc[:, 'rt_resp'], bins=bins, density=True,)
            hist_sim = hist_sim * fraction_yes_sim
            ax.bar(edges[:-1], hist, width=np.diff(edges)[0], align='edge', 
                   color='magenta', alpha=0.5, linewidth=0,)
            # ax.plot(edges_sim[:-1], hist_sim, color='k', lw=1)
            ax.step(edges_sim[:-1]+np.diff(edges)[0], hist_sim, color='black', lw=1)
            # ax.hist(hist, edges, histtype='stepfilled', color='magenta', alpha=0.5, label='response')
            # ax.hist(hist_sim, edges_sim, histtype='step', color='k',)
            no_height = (1 - fraction_yes) / bar_width 
            no_height_sim = (1 - fraction_yes_sim) / bar_width 
            ax.bar(x=-1.5, height=no_height, width=bar_width, alpha=0.5, color='cyan', align='center')
            ax.hlines(y=no_height_sim, xmin=-2, xmax=-1, lw=0.5, colors='black',)
            ax.vlines(x=-2, ymin=0, ymax=no_height_sim, lw=0.5, colors='black')
            ax.vlines(x=-1, ymin=0, ymax=no_height_sim, lw=0.5, colors='black')
        else:
            N, bins, patches = ax.hist(df.loc[:, 'rt_resp'], bins=bins, 
                                   density=True, color='magenta', alpha=0.5)       
            for bin_size, bin, patch in zip(N, bins, patches):
                if bin < 0:
                    plt.setp(patch, 'facecolor', 'cyan')
            ax.hist(df_sim.loc[:, 'rt_resp'], bins=bins, density=True, 
                    histtype='step', color='k', alpha=1, label=None) 
        ax.set_title('P(bias)={}'.format(round(df.loc[:, 'response'].mean(), 3),))
        ax.set_xlabel('RT (s)')
        ax.set_ylabel('Trials (prob. dens.)')
        plt_nr += 1
        
        # condition response plots:
        ax = fig.add_subplot(nr_subjects,4,plt_nr)
        df.loc[:,'rt_bin'] = pd.qcut(df['rt'], quantiles, labels=False)
        d = df.groupby(['rt_bin']).mean().reset_index()
        ax.errorbar(d.loc[:, "rt"], d.loc[:, "response"], fmt='-o', color='orange', markersize=10)
        if df_sim is not None:
            df_sim.loc[:,'rt_bin'] = pd.qcut(df_sim['rt'], quantiles, labels=False)
            d = df_sim.groupby(['rt_bin']).mean().reset_index()
            ax.errorbar(d.loc[:, "rt"], d.loc[:, "response"], fmt='x', color='k', markersize=6)
        if xlim:
            ax.set_xlim(xlim)
        ax.set_ylim(0,1)
        ax.set_title('Conditional response')
        ax.set_xlabel('RT (quantiles)')
        ax.set_ylabel('P(bias)')
        plt_nr += 1
        
    sns.despine(offset=3, trim=True)
    plt.tight_layout()

    return fig 