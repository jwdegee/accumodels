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

class OneAccumulatorNoise(ddm.models.Noise):
    name = "Noise"
    required_parameters = ["noise"]
    required_conditions = ['start', 'duration']
    def get_noise(self, t, conditions, **kwargs):
        
        if t < (conditions['start']+conditions['duration']):
            return self.noise
        else:
            return 0

class OneAccumulatorStartingPoint(InitialCondition):
    name = 'A starting point.'
    required_parameters = ['a0', 'a1', 'sz']
    required_conditions = ['choice_p']
    def get_IC(self, x, dx, conditions):
        
        a = getattr(self, 'a{}'.format(conditions['choice_p']))
        sz = self.sz
        x0 = 10 - a
        # assert abs(x0) + abs(sz) < np.max(x), \
        #     "Invalid x0 and sz: distribution goes past simulation boundaries"
        pdf = np.zeros(len(x))
        ind = (x>=x0)&(x<=(x0+sz))
        if sum(ind) <= 1:
            pdf[np.where(x>=x0)[0][0]] = 1
        else:
            pdf[ind] = 1
        try:
            pdf = pdf/np.sum(pdf)
        except Exception as e:
            print('x0 = {}'.format(x0))
            print('sz = {}'.format(sz))
            print(x)
            print(pdf)
            print(e)
            raise ValueError('A very specific bad thing happened.')
        return pdf

class OneAccumulatorDrift(ddm.models.Drift):
    name = 'Drift'
    required_parameters = ['a0', 'a1', 'v', 'b', 'k', 'tt']
    required_conditions = ['start', 'duration', 'choice_p']
    def get_drift(self, x, t, conditions, **kwargs):
        a = getattr(self, 'a{}'.format(conditions['choice_p']))
        v = self.v
        b = self.b
        k = self.k
        k_target = 10 - a
        non_dec_time = self.tt
        if t < (conditions['start'] + non_dec_time):
            return b - (k * (x-k_target))
        elif (t >= (conditions['start'] + non_dec_time)) & (t < (conditions['start']+conditions['duration'])):
            return v + b - (k * (x-k_target))
        else:
            return 0

class OneAccumulatorNonDecisionTime(Overlay):
    name = 'Non-decision time'
    required_parameters = ['t']
    required_conditions = []
    def apply(self, solution):
        # Unpack solution object
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        
        non_dec_time = self.t
        shifts = int(non_dec_time/m.dt) # truncate
        
        # Shift the distribution
        newcorr = np.zeros(corr.shape, dtype=corr.dtype)
        newerr = np.zeros(err.shape, dtype=err.dtype)
        if shifts > 0:
            newcorr[shifts:] = corr[:-shifts]
            # newerr[shifts:] = err[:-shifts]
        elif shifts < 0:
            newcorr[:shifts] = corr[-shifts:]
            # newerr[:shifts] = err[-shifts:]
        else:
            newcorr = corr
            # newerr = err
        newerr = err
        return Solution(newcorr, newerr, m, cond, undec)

class OneAccumulatorOverlayGaussMixture(Overlay):
    """An exponential mixture distribution.
    The output distribution should be pmixturecoef*100 percent exponential
    distribution and (1-umixturecoef)*100 percent of the distribution
    to which this overlay is applied.
    A mixture with the exponential distribution can be used to confer
    robustness when fitting using likelihood.
    Note that this is called OverlayPoissonMixture and not
    OverlayExponentialMixture because the exponential distribution is
    formed from a Poisson process, i.e. modeling a uniform lapse rate.
    Example usage:
      | overlay = OverlayPoissonMixture(pmixturecoef=.02, rate=1)
    """
    name = "Poisson distribution mixture model (lapse rate)"
    required_parameters = ["mixture", "mu", "sigma"]
    required_conditions = []
    def apply(self, solution):
        
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        evolution = solution.evolution

        mixturecoef = self.mixture
        mu = self.mu
        sigma = self.sigma
        
        assert mixturecoef >= 0 and mixturecoef <= 1
        assert isinstance(solution, Solution)

        # To make this work with undecided probability, we need to
        # normalize by the sum of the decided density.  That way, this
        # function will never touch the undecided pieces.
        
        norm = np.sum(corr) #+ np.sum(err)
        X = m.dt * np.arange(0, len(corr))
        Y = sp.stats.norm.pdf(X, loc=mu, scale=sigma)
        Y /= np.sum(Y)
        corr = corr*(1-mixturecoef) + mixturecoef*Y*norm # Assume numpy ndarrays, not lists

        return Solution(corr, err, m, cond, undec, evolution)

class OneAccumulatorOverlayEvidenceLapse(Overlay):
    """An exponential mixture distribution.
    The output distribution should be pmixturecoef*100 percent exponential
    distribution and (1-umixturecoef)*100 percent of the distribution
    to which this overlay is applied.
    A mixture with the exponential distribution can be used to confer
    robustness when fitting using likelihood.
    Note that this is called OverlayPoissonMixture and not
    OverlayExponentialMixture because the exponential distribution is
    formed from a Poisson process, i.e. modeling a uniform lapse rate.
    Example usage:
      | overlay = OverlayPoissonMixture(pmixturecoef=.02, rate=1)
    """
    name = "evidence lapse"
    required_parameters = ['lapse', 'lapse_slope']
    required_conditions = ['start', 'duration']
    def apply(self, solution):
        
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        evolution = solution.evolution
        
        time = cond['start']
        lapse_offset = self.lapse
        lapse_slope = self.lapse_slope
        lapse = lapse_offset + (time * lapse_slope)

        # make sure that lapse is bounded between 0 and 1:
        lapse = np.clip(lapse, 0, 1)

        # assert:
        assert isinstance(solution, Solution)

        # check what corr would look like with drift rate == 0
        # ----------------------------------------------------

        m2 = copy.copy(m)
        
        # set depenencies
        dependencies = m.dependencies
        for i in range(len(dependencies)):
            if dependencies[i].depname == 'Overlay':
                overlays = dependencies[i].overlays
                for j in range(len(overlays)):
                    if overlays[j].name == 'evidence lapse':
                        overlays.pop(j)
                dependencies[i] = OverlayChain(overlays=overlays)
        m2.dependencies = dependencies

        # set parameters:
        param_names = m.get_model_parameter_names()
        param_values = m.get_model_parameters()
        for i in range(len(param_values)):
            if param_names[i][0] == 'v':
                param_values[i] = Fitted(0)
        m2.set_model_parameters(param_values)

        # solve:
        solution2 = m2.solve(cond)
        corr2 = solution2.corr

        # ----------------------------------------------------
        
        # update corr:
        corr = (corr*(1-lapse)) + (lapse*corr2) # Assume numpy ndarrays, not lists

        return Solution(corr, err, m, cond, undec, evolution)

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

def make_drift(sample, drift_bias, leak, v_depends_on=[None], b_depends_on=[None], k_depends_on=[None]):

    if 'run' in v_depends_on:
        run_regressor = True
        v_depends_on.remove('run')
        print(v_depends_on)
    else:
        run_regressor = False

    v_names, v_unique_conditions = get_param_names(sample=sample, depends_on=v_depends_on, param='v')
    if drift_bias:
        b_names, b_unique_conditions = get_param_names(sample=sample, depends_on=b_depends_on, param='b')
    else:
        b_names = []
    if leak:
        k_names, k_unique_conditions = get_param_names(sample=sample, depends_on=k_depends_on, param='k')
    else:
        k_names = []

    class DriftStimulusCoding(ddm.models.Drift):
        name = 'Drift'
        required_parameters = v_names + b_names + k_names
        required_conditions = ['stimulus']

        if (v_depends_on is not None):
            required_conditions = list(set(required_conditions+v_depends_on))
        if (b_depends_on is not None):
            required_conditions = list(set(required_conditions+b_depends_on))
        if (k_depends_on is not None):
            required_conditions = list(set(required_conditions+k_depends_on))
        
        if run_regressor:
            required_conditions = list(set(required_conditions+['run']))
            print(required_conditions)

        def get_drift(self, x, conditions, **kwargs):
            
            # v param:
            if v_depends_on is None:
                v_param = self.v
            elif len(v_unique_conditions) == 1:
                v_param = getattr(self, 'v{}'.format(conditions[v_depends_on[0]]))
            elif len(v_unique_conditions) == 2:
                v_param = getattr(self, 'v{}.{}'.format(conditions[v_depends_on[0]],conditions[v_depends_on[1]]))

            if drift_bias:
                # b param:
                if b_depends_on is None:
                    b_param = self.b
                elif len(b_unique_conditions) == 1:
                    b_param = getattr(self, 'b{}'.format(conditions[b_depends_on[0]]))
                elif len(b_unique_conditions) == 2:
                    b_param = getattr(self, 'b{}.{}'.format(conditions[b_depends_on[0]],conditions[b_depends_on[1]]))
            else:
                b_param = 0

            if leak:
                # b param:
                if k_depends_on is None:
                    k_param = self.k
                elif len(k_unique_conditions) == 1:
                    k_param = getattr(self, 'k{}'.format(conditions[k_depends_on[0]]))
                elif len(b_unique_conditions) == 2:
                    k_param = getattr(self, 'k{}.{}'.format(conditions[k_depends_on[0]],conditions[k_depends_on[1]]))
            else:
                k_param = 0
            
            if run_regressor:
                run = conditions['run']
            else:
                run = 1

            stim = conditions['stimulus']

            # return:
            return (v_param * stim * run) + b_param - (k_param * x)

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


def make_model_one_accumulator(model_settings):

    # parameters:
    ranges = {
            # 'v':(0.99,1.01),               # drift rate
            # 'b':(-0.21,-0.19),             # drift bias
            # 'k':(2.39,2.41),               # leak
            # 'z':(0.75,0.999),              # starting point --> translates into bound heigth 0-5
            'v':(0,10),                      # drift rate
            'b':(-5,5),                      # drift bias
            'k':(-10,10),                    # leak
            'a':(0.5,1.5),                   # bound
            't':(0,0.1),                     # non-decision time
            'sz':(0,1),                      # starting point variability
            'lapse':(0.001,0.999),           # lapse rate
            'lapse_slope':(0,0.1),           # lapse rate slope
            'mixture':(0.001,0.999),         # mixture rate
            }
    
    
    if (model_settings['depends_on']['a'] is None):
        a0_value = Fittable(minval=ranges['a'][0], maxval=ranges['a'][1], default=1)
        a1_value = a0_value
    else:
        a0_value = Fittable(minval=ranges['a'][0], maxval=ranges['a'][1], default=1)
        a1_value = Fittable(minval=ranges['a'][0], maxval=ranges['a'][1], default=1)
    
    t_value = Fittable(minval=ranges['t'][0], maxval=ranges['t'][1], default=1)
    v_value = Fittable(minval=ranges['v'][0], maxval=ranges['v'][1], default=1)
    k_value = Fittable(minval=ranges['k'][0], maxval=ranges['k'][1], default=1)
    b_value = 0

    if model_settings['lapse']:
        lapse_value = Fittable(minval=ranges['lapse'][0], maxval=ranges['lapse'][1], default=0.5)
    else:
        lapse_value = 0

    if model_settings['lapse_slope']:
        lapse_slope_value = Fittable(minval=ranges['lapse_slope'][0], maxval=ranges['lapse_slope'][1], default=0.05)
    else:
        lapse_slope_value = 0

    if model_settings['non_dec_t']:
        tt_value = Fittable(minval=ranges['t'][0], maxval=ranges['t'][1], default=0.1)
    else:
        tt_value = 0

    if model_settings['sz']:
        sz_value = Fittable(minval=ranges['sz'][0], maxval=ranges['sz'][1], default=0.1)
    else:
        sz_value = 0

    if model_settings['mixture']:
        mixture_value = Fittable(minval=ranges['mixture'][0], maxval=ranges['mixture'][1], default=0.1)
        mu_value = Fittable(minval=0, maxval=0.5, default=0.2)
        sigma_value = Fittable(minval=0.001, maxval=0.5, default=0.05)
    else:
        mixture_value = 0
        mu_value = 0
        sigma_value = 1

    # components:
    starting_point_components = {'a0':a0_value, 'a1':a1_value, 'sz':sz_value}
    drift_components = {
                        'v':v_value,
                        'k':k_value,
                        'b':b_value,
                        'a0':a0_value,
                        'a1':a1_value,
                        'tt':tt_value,
                        }
    mixture_components = {'mixture':mixture_value, 'mu':mu_value, 'sigma':sigma_value}
    lapse_components = {'lapse':lapse_value, 'lapse_slope':lapse_slope_value}
    non_dec_components = {'t':t_value, 't':t_value}

    # build model:
    from ddm.models import BoundConstant, OverlayChain
    model = Model(name='one accumulator model',
                IC=OneAccumulatorStartingPoint(**starting_point_components),
                drift=OneAccumulatorDrift(**drift_components),
                bound=BoundConstant(B=10),
                overlay=OverlayChain(overlays=[
                                                OneAccumulatorNonDecisionTime(**non_dec_components),
                                                OneAccumulatorOverlayGaussMixture(**mixture_components),
                                                OneAccumulatorOverlayEvidenceLapse(**lapse_components),
                                                ]),
                noise=OneAccumulatorNoise(noise=1),
                dx=model_settings['dx'], dt=model_settings['dt'], T_dur=model_settings['T_dur'])
    
    return model

def fit_model_one_accumulator(df, model_settings):

    from ddm import set_N_cpus
    set_N_cpus(14)

    # sample:
    sample = Sample.from_pandas_dataframe(df=df, rt_column_name='rt', correct_column_name='response')

    # make model
    model = make_model_one_accumulator(model_settings=model_settings)

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


def make_model(sample, model_settings):
    
    # fitting parameters:
    T_dur = model_settings['T_dur']
    if 'dx' in model_settings:
        dx = model_settings['dx']
    else:
        dx = 0.001
    if 'dt' in model_settings:
        dt = model_settings['dt']
    else:
        dt = 0.01

    # model components:
    z = make_z(sample=sample, 
                z_depends_on=model_settings['depends_on']['z'])
    drift = make_drift(sample=sample, 
                        drift_bias=model_settings['drift_bias'], 
                        leak=model_settings['leak'], 
                        v_depends_on=model_settings['depends_on']['v'], 
                        b_depends_on=model_settings['depends_on']['b'],
                        k_depends_on=model_settings['depends_on']['k'])
    a = make_a(sample=sample, 
                urgency=model_settings['urgency'], 
                a_depends_on=model_settings['depends_on']['a'], 
                u_depends_on=model_settings['depends_on']['u'])
    t = make_t(sample=sample, 
                t_depends_on=model_settings['depends_on']['t'])
    
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
    if model_settings['start_bias']:
        initial_condition = z(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in z.required_parameters})
    else:
        initial_condition = z(**{'z':0.5})
    model = Model(name='stimulus coding model / collapsing bound',
                IC=initial_condition,
                drift=drift(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in drift.required_parameters}),
                bound=a(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in a.required_parameters}),
                overlay=OverlayChain(overlays=[t(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in t.required_parameters}),
                                                # OverlayUniformMixture(umixturecoef=0)]),
                                                OverlayPoissonMixture(pmixturecoef=.05, rate=1)]),
                noise=NoiseConstant(noise=1),
                dx=dx, dt=dt, T_dur=T_dur)
    return model

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