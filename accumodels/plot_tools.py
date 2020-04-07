#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from IPython import embed as shell

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

def summary_plot_group(df_group, df_sim_group=None, quantiles=[0, 0.1, 0.3, 0.5, 0.7, 0.9,], step_size=0.05, xlim=None):

    # # remove NaNs:
    # df = df.loc[~pd.isna(df.rt),:]
    # if df_sim is not None:
    #     df_sim = df_sim.loc[~pd.isna(df_sim.rt),:]

    # step_size = 0.125

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
    if xlim:
        ax.set_xlim((-xlim[1],xlim[1]))
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
    if xlim:
        ax.set_xlim((-xlim[1],xlim[1]))
    ax.set_title('P(bias)={} +-{}'.format(
                                round(df_group.groupby('subj_idx')['response'].mean().mean(), 3),
                                round(df_group.groupby('subj_idx')['response'].mean().std(), 3),))
    ax.set_xlabel('RT (s)')
    ax.set_ylabel('Trials (prob. dens.)')

    # conditional response plots:
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
    plt.axhline(0.5, lw=0.5, color='k')
    if xlim:
        ax.set_xlim(xlim)
    ax.set_ylim(0, 0.9)
    ax.set_title('Conditional response')
    ax.set_xlabel('RT (quantiles)')
    ax.set_ylabel('P(bias)')

    sns.despine(offset=3, trim=True)
    plt.tight_layout()

    return fig

def summary_plot(df, quantiles, mean_correct, mean_response, xlim=None):

    fig = plt.figure(figsize=(2,6))

    df = df.loc[(df.rt > (df.rt.mean()-(4*df.rt.std()))) & (df.rt < (df.rt.mean()+(4*df.rt.std())))]

    # rt distributions:
    ax = fig.add_subplot(3,1,1)
    ax.hist(df.loc[(df.correct==0), 'rt']*-1.0, histtype='stepfilled', color='red', alpha=0.5, bins=10)
    ax.hist(df.loc[(df.correct==1), 'rt'], histtype='stepfilled', color='forestgreen', alpha=0.5, bins=10)
    ax.set_xlim(-2,2)
    ax.set_title('P(bias)={}; P(correct)={}'.format(round(df.loc[:, 'response'].mean(), 3), round(df.loc[:, 'correct'].mean(), 3)))
    ax.set_xlabel('RT (s)')
    ax.set_ylabel('Trials (#)')

    # condition accuracy plots:
    ax = fig.add_subplot(3,1,2)
    plt.axhline(mean_correct, lw=0.5, color='k')
    df.loc[:,'rt_bin'] = pd.qcut(df['rt'], quantiles, labels=False)
    d = df.groupby(['rt_bin']).mean().reset_index()
    # ax.plot(np.array(quantiles)[1:], np.array(df.groupby('rt_bin').mean()['correct']), color='k')
    # plt.xticks(np.array(quantiles)[1:], list(np.array(quantiles)[1:]))
    ax.errorbar(d.loc[:, "rt"], d.loc[:, "correct"], fmt='-o', color='k', markersize=5)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_ylim(0.4, 1)
    ax.set_title('Conditional accuracy')
    ax.set_xlabel('RT (quantiles)')
    ax.set_ylabel('P(correct)')

    # condition response plots:
    ax = fig.add_subplot(3,1,3)
    df.loc[:,'rt_bin'] = pd.qcut(df['rt'], quantiles, labels=False)
    plt.axhline(mean_response, lw=0.5, color='k')
    d = df.groupby(['rt_bin']).mean().reset_index()
    # ax.plot(np.array(quantiles)[1:], np.array(df.groupby('rt_bin').mean()['response']), color='k')
    # plt.xticks(np.array(quantiles)[1:], list(np.array(quantiles)[1:]))
    ax.errorbar(d.loc[:, "rt"], d.loc[:, "response"], fmt='-o', color='k', markersize=5)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_ylim(0.4,1)
    ax.set_title('Conditional response')
    ax.set_xlabel('RT (quantiles)')
    ax.set_ylabel('P(bias)')
    sns.despine(offset=10, trim=True)
    plt.tight_layout()

    return fig

def conditional_response_plot(df, quantiles, mean_response, y='response', xlim=[0.1,0.6], ylim=[0.4,0.9], cmap="Blues"):
    

    fig = plt.figure(figsize=(1.5,1.5))
    ax = fig.add_subplot(1,1,1)
    
    plt.axhline(mean_response, xmin=xlim[0]-0.1, xmax=xlim[1]+0.1, lw=0.5, color='k')
    df.loc[:,'rt_bin'] = pd.qcut(df['rt'], quantiles, labels=False)
    d = df.groupby(['subj_idx', 'rt_bin']).mean().reset_index()
    # for s, a in zip(np.unique(d["subj_idx"]), [0.1, 0.5, 0.9]):
    #     ax.plot(np.array(quantiles)[1:], d.loc[d["subj_idx"]==s, "response"], color='k', alpha=a)
    # plt.xticks(np.array(quantiles)[1:], list(np.array(quantiles)[1:]))

    colors = sns.color_palette(cmap, len(np.unique(df['subj_idx'])))
    for s, c in zip(np.unique(d["subj_idx"]), colors):
        ax.errorbar(d.loc[d["subj_idx"]==s, "rt"], d.loc[d["subj_idx"]==s, y], fmt='-o', color=c, markersize=5)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    means = df.groupby(['subj_idx']).mean()
    means = [round(m,2) for m in means['correct']] + [round(m,2) for m in means['response']]

    ax.set_title('P(corr.)={}, {}, {}\nP(bias)={}, {}, {}'.format(*means))
    ax.set_xlabel('RT (s)')
    ax.set_ylabel('P({})'.format(y))
    sns.despine(trim=True, offset=2)
    plt.tight_layout()
    
    return fig


def conditional_history_plot(df, quantiles, mean_response, xlim=[0.1, 0.6], cmap="Blues"):
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(1, 1, 1)
    plt.axhline(mean_response, xmin=xlim[0] - 0.1, xmax=xlim[1] + 0.1, lw=0.5, color='k')
    df.loc[:, 'rt_bin'] = pd.qcut(df['rt'], quantiles, labels=False)
    d = df.groupby(['subj_idx', 'rt_bin']).mean().reset_index()
    # for s, a in zip(np.unique(d["subj_idx"]), [0.1, 0.5, 0.9]):
    #     ax.plot(np.array(quantiles)[1:], d.loc[d["subj_idx"]==s, "response"], color='k', alpha=a)
    # plt.xticks(np.array(quantiles)[1:], list(np.array(quantiles)[1:]))

    colors = sns.color_palette(cmap, len(np.unique(df['subj_idx'])))
    for s, c in zip(np.unique(d["subj_idx"]), colors):
        ax.errorbar(d.loc[d["subj_idx"] == s, "rt"], d.loc[d["subj_idx"] == s, "repeat"], fmt='-o', color=c,
                    markersize=5)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_ylim(0.4, 1)
    ax.set_title('P(correct) = {}\nP(repeat) = {}'.format(
        round(df.loc[:, 'correct'].mean(), 2),
        round(df.loc[:, 'repeat'].mean(), 2),
    ))
    ax.set_xlabel('RT (s)')
    ax.set_ylabel('P(repeat)')
    sns.despine(trim=True, offset=5)
    plt.tight_layout()

    return fig

def traces_plot(df, x1, x2, a, ndt):
    
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(1,1,1)
    t = 0
    plotted = 0
    while plotted < 10:
        if df.iloc[t]['response'] == 1:
            x1_trace = x1.iloc[t]
            x2_trace = x2.iloc[t]
            cutoff = df.iloc[t]['rt'] - ndt
            x1_trace.loc[x1.columns > cutoff] = np.NaN
            x2_trace.loc[x1.columns > cutoff] = np.NaN
            ax.plot(x1_trace, lw=0.5, color='r')
            ax.plot(x2_trace, lw=0.5, color='b', alpha=0.5)
            plotted += 1
        t += 1
    
    plt.axhline(a, lw=1, color='green')
    ax.set_xlabel('Time')
    ax.set_ylabel('Activity (a.u.)')
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    return fig
