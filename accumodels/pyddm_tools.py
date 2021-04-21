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
from ddm.models.loss import LossFunction
from ddm.functions import fit_adjust_model, display_model
from ddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture, OverlayUniformMixture, InitialCondition, ICPoint, ICPointSourceCenter, LossBIC, LossRobustBIC, LossBICGonogo, LossRobustBICGonogo
# from ddm import set_N_cpus

from IPython import embed as shell

# set_N_cpus(16)

# class StartingPoint(InitialCondition):
#     name = 'A starting point.'
#     required_parameters = ['a']
#     def get_IC(self, x, dx, conditions):
#         a = self.a
#         z = 10 - a
#         pdf = np.zeros(len(x))
#         pdf[np.where(x>=z)[0][0]] = 1
#         return pdf

class StartingPoint_reward(InitialCondition):
    name = 'A starting point.'
    required_parameters = ['a0', 'a1']
    required_conditions = ['reward']
    def get_IC(self, x, dx, conditions):
        a = getattr(self, 'a{}'.format(conditions['reward']))
        start = 10 - a
        pdf = np.zeros(len(x))
        pdf[np.where(x>=start)[0][0]] = 1
        return pdf

def make_z_one_accumulator(sample, a_depends_on=[None]):
    
    a_names, a_unique_conditions = get_param_names(sample=sample, depends_on=a_depends_on, param='a')

    class StartingPoint(InitialCondition):
        name = 'A starting point.'
        required_parameters = a_names
        if not a_depends_on is None:
            required_conditions = a_depends_on.copy()
        def get_IC(self, x, dx, conditions):
            pdf = np.zeros(len(x))
            if a_depends_on is None:
                a_param = self.a
            elif len(a_unique_conditions) == 1:
                a_param = getattr(self, 'a{}'.format(conditions[a_depends_on[0]]))
            elif len(a_unique_conditions) == 2:
                a_param = getattr(self, 'a{}.{}'.format(conditions[a_depends_on[0]], conditions[a_depends_on[1]]))
            z_param = 10 - a_param
            pdf[int(len(pdf)*z_param)] = 1
            return pdf
    return StartingPoint


# class DriftPulse(ddm.models.Drift):
#     name = 'Drift'
#     required_parameters = ['v', 'b', 'k', 'a']
#     required_conditions = ['start', 'duration']
#     def get_drift(self, x, t, conditions, **kwargs):
#         a = self.a
#         v = self.v
#         b = self.b
#         k = self.k
#         k_target = 10 - a
#         if t < conditions['start']:
#             return b - (k * (x-k_target))
#         elif (t >= conditions['start']) & (t < (conditions['start']+conditions['duration'])):
#             return v + b - (k * (x-k_target))
#         else:
#             return 0


class DriftPulse_reward(ddm.models.Drift):
    name = 'Drift'
    required_parameters = ['v0', 'b0', 'k0', 'a0', 'v1', 'b1', 'k1', 'a1']
    required_conditions = ['start', 'duration', 'reward']
    def get_drift(self, x, t, conditions, **kwargs):
        a = getattr(self, 'a{}'.format(conditions['reward']))
        v = getattr(self, 'v{}'.format(conditions['reward']))
        b = getattr(self, 'b{}'.format(conditions['reward']))
        k = getattr(self, 'k{}'.format(conditions['reward']))
        k_target = 10 - a
        if t < conditions['start']:
            return b - (k * (x-k_target))
        elif (t >= conditions['start']) & (t < (conditions['start']+conditions['duration'])):
            return v + b - (k * (x-k_target))
        else:
            return 0

def make_drift_one_accumulator(sample, drift_bias, leak, v_depends_on=[None], b_depends_on=[None], k_depends_on=[None], a_depends_on=[None]):
    
    v_names, v_unique_conditions = get_param_names(sample=sample, depends_on=v_depends_on, param='v')
    a_names, a_unique_conditions = get_param_names(sample=sample, depends_on=a_depends_on, param='a')
    if drift_bias:
        b_names, b_unique_conditions = get_param_names(sample=sample, depends_on=b_depends_on, param='b')
    else:
        b_names = []
    if leak:
        k_names, k_unique_conditions = get_param_names(sample=sample, depends_on=k_depends_on, param='k')
    else:
        k_names = []
    
    class DriftPulse(ddm.models.Drift):
        name = 'Drift'
        required_parameters = v_names + b_names + k_names + a_names
        required_conditions = ['start', 'duration']
        if (v_depends_on is not None):
            required_conditions = list(set(required_conditions+v_depends_on))
        if (b_depends_on is not None):
            required_conditions = list(set(required_conditions+b_depends_on))
        if (k_depends_on is not None):
            required_conditions = list(set(required_conditions+k_depends_on))
        if (a_depends_on is not None):
            required_conditions = list(set(required_conditions+a_depends_on))

        def get_drift(self, x, t, conditions, **kwargs):
            
            # v param:
            if v_depends_on is None:
                v_param = self.v
            elif len(v_unique_conditions) == 1:
                v_param = getattr(self, 'v{}'.format(conditions[v_depends_on[0]]))
            elif len(v_unique_conditions) == 2:
                v_param = getattr(self, 'v{}.{}'.format(conditions[v_depends_on[0]],conditions[v_depends_on[1]]))

            # a param:
            if a_depends_on is None:
                a_param = self.a
            elif len(a_unique_conditions) == 1:
                a_param = getattr(self, 'a{}'.format(conditions[a_depends_on[0]]))
            elif len(v_unique_conditions) == 2:
                a_param = getattr(self, 'a{}.{}'.format(conditions[a_depends_on[0]],conditions[a_depends_on[1]]))

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

            # return:
            k_target = 10 - a_param

            print(k_target)
            # print(t)

            # shell()

            if t < conditions['start']:
                return b_param - (k_param * (x-k_target))
            elif (t >= conditions['start']) & (t < (conditions['start']+conditions['duration'])):
                return v_param + b_param - (k_param * (x-k_target))
            else:
                return 0

    return DriftPulse

class NoisePulse(ddm.models.Noise):
    name = "Noise"
    required_parameters = ["noise"]
    required_conditions = ['start', 'duration']
    def get_noise(self, t, conditions, **kwargs):
        
        if t < (conditions['start']+conditions['duration']):
            return self.noise
        else:
            return 0

class Bound(Bound):
    name = 'Hyperbolic collapsing bounds'
    required_parameters = ['a']
    required_conditions = []

    def get_bound(self, t, conditions, **kwargs):
        return self.a

class NonDecisionTime(Overlay):
    name = 'Non-decision time'
    required_parameters = ['t']
    def apply(self, solution):
        # Unpack solution object
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        shifts = int(self.t/m.dt) # truncate
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

            if leak:
                # b param:
                if k_depends_on is None:
                    k_param = self.k
                elif len(k_unique_conditions) == 1:
                    k_param = getattr(self, 'k{}'.format(conditions[k_depends_on[0]]))
                elif len(b_unique_conditions) == 2:
                    k_param = getattr(self, 'k{}.{}'.format(conditions[k_depends_on[0]],conditions[k_depends_on[1]]))

            # return:
            if drift_bias & leak:
                return (v_param * conditions['stimulus']) + b_param - (k_param * x)
            elif drift_bias:
                return (v_param * conditions['stimulus']) + b_param
            elif leak:
                return (v_param * conditions['stimulus']) - (k_param * x)
            else:
                return (v_param * conditions['stimulus'])
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


def make_model_one_accumulator(sample, model_settings):

    # # components:
    # z = make_z_one_accumulator(sample=sample, 
    #             a_depends_on=model_settings['depends_on']['a'])
    # drift = make_drift_one_accumulator(sample=sample, 
    #                 drift_bias=model_settings['drift_bias'], 
    #                 leak=model_settings['leak'], 
    #                 v_depends_on=model_settings['depends_on']['v'],
    #                 b_depends_on=model_settings['depends_on']['b'],
    #                 k_depends_on=model_settings['depends_on']['k'],
    #                 a_depends_on=model_settings['depends_on']['v'],)
    # bound = BoundConstant
    # t = NonDecisionTime
    # n = NoisePulse(noise=1)

    # # limits:
    # ranges = {

    #         # 'v':(0.99,1.01),               # drift rate
    #         # 'b':(-0.21,-0.19),             # drift bias
    #         # 'k':(2.39,2.41),               # leak
    #         # 'z':(0.75,0.999),              # starting point --> translates into bound heigth 0-5
    #         'v':(0,5),                     # drift rate
    #         'b':(-5,5),                    # drift bias
    #         'k':(0,5),                     # leak
    #         'a':(0.1,5),                   # bound
    #         't':(0,.1),                    # non-decision time
    #         }
    
    # # initialize params:
    # a_params = {param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in z.required_parameters}
    # drift_params = {param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in drift.required_parameters if not 'a' in param}
    # drift_params = {**drift_params, **a_params}

    # # put together:
    # model = Model(name='one accumulator model',
    #             IC=z(**a_params),
    #             drift=drift(**drift_params),
    #             bound=bound(B=10),
    #             overlay=OverlayChain(overlays=[t(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in t.required_parameters}),
    #                                             # OverlayUniformMixture(umixturecoef=0)
    #                                             # OverlayUniformMixture(umixturecoef=0.01)
    #                                             OverlayPoissonMixture(pmixturecoef=.02, rate=1)
    #                                             ]),
    #             noise=n,
    #             dx=.01, dt=.01, T_dur=T_dur)

    # parameters:
    ranges = {
            # 'v':(0.99,1.01),               # drift rate
            # 'b':(-0.21,-0.19),             # drift bias
            # 'k':(2.39,2.41),               # leak
            # 'z':(0.75,0.999),              # starting point --> translates into bound heigth 0-5
            'v':(0,5),                       # drift rate
            'b':(-5,5),                      # drift bias
            'k':(0,5),                       # leak
            'a':(0.1,2),                     # bound
            't':(0,.1),                      # non-decision time
            }
    a0_value = Fittable(minval=ranges['a'][0], maxval=ranges['a'][1], default=1)
    a1_value = Fittable(minval=ranges['a'][0], maxval=ranges['a'][1], default=1)
    v0_value = Fittable(minval=ranges['v'][0], maxval=ranges['a'][1], default=1)
    v1_value = Fittable(minval=ranges['v'][0], maxval=ranges['a'][1], default=1)

    if model_settings['drift_bias'] & (model_settings['depends_on']['b'] == ['reward']):
        b0_value = Fittable(minval=ranges['b'][0], maxval=ranges['b'][1], default=0)
        b1_value = Fittable(minval=ranges['b'][0], maxval=ranges['b'][1], default=0)
    elif model_settings['drift_bias'] & (model_settings['depends_on']['b'] is None):
        b0_value = Fittable(minval=ranges['b'][0], maxval=ranges['b'][1], default=0)
        b1_value = b0_value
    else:
        b0_value = 0
        b1_value = 0

    if model_settings['leak'] & (model_settings['depends_on']['k'] == ['reward']):
        k0_value = Fittable(minval=ranges['k'][0], maxval=ranges['k'][1], default=1)
        k1_value = Fittable(minval=ranges['k'][0], maxval=ranges['k'][1], default=1)
    elif model_settings['leak'] & (model_settings['depends_on']['k'] is None):
        k0_value = Fittable(minval=ranges['k'][0], maxval=ranges['k'][1], default=1)
        k1_value = k0_value
    else:
        k0_value = 0
        k1_value = 0


    
    # components:
    starting_point_components = {'a0':a0_value, 'a1':a1_value}
    drift_components = {
                        'v0':v0_value,
                        'v1':v1_value,
                        'k0':k0_value,
                        'k1':k1_value,
                        'b0':b0_value, 
                        'b1':b1_value,
                        'a0':a0_value, 
                        'a1':a1_value,}

    # build model:
    from ddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture, OverlayUniformMixture, InitialCondition, ICPoint, ICPointSourceCenter, LossBIC
    bound = BoundConstant
    a = StartingPoint_reward
    drift = DriftPulse_reward
    t = NonDecisionTime
    n = NoisePulse(noise=1)
    model = Model(name='one accumulator model',
                IC=a(**starting_point_components),
                drift=drift(**drift_components),
                bound=bound(B=10),
                overlay=OverlayChain(overlays=[t(t=0),
                                                # t(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in t.required_parameters}),
                                                # OverlayUniformMixture(umixturecoef=0)
                                                # OverlayUniformMixture(umixturecoef=0.01)
                                                OverlayPoissonMixture(pmixturecoef=.1, rate=1)
                                                ]),
                # overlay=OverlayChain(overlays=[t(t=0.1), OverlayUniformMixture(umixturecoef=0.01)]),
                noise=n,
                dx=model_settings['dx'], dt=model_settings['dt'], T_dur=model_settings['T_dur'])

    # # fit:
    # # model_fit = fit_adjust_model(sample=sample, model=model, lossfunction=LossLikelihood)
    # try:
        
    #     model_fit = fit_adjust_model(sample=sample, model=model, lossfunction=LossLikelihoodGonogo, fitting_method="differential_evolution")
        
    #     # print('model: {}'.format(model_fit.solve({'stimulus':0}).prob_correct()))
    #     # print('data: {}'.format(df.loc[df['stimulus']==0, 'response'].mean()))
    #     # print()
    #     # print('model: {}'.format(model_fit.solve({'stimulus':1}).prob_correct()))
    #     # print('data: {}'.format(df.loc[df['stimulus']==1, 'response'].mean()))

    #     # # plot:
    #     # ddm.plot.plot_fit_diagnostics(model=model_fit, sample=sample)

    #     # get params:
    #     params = pd.DataFrame(np.atleast_2d([p.real for p in model_fit.get_model_parameters()]), columns=model_fit.get_model_parameter_names())
    #     params['bic'] = model_fit.fitresult.value()
    # except Exception as e: 
        
    #     print(e)
    #     params = pd.DataFrame(np.atleast_2d([np.nan, np.nan, np.nan, np.nan]), columns=model.get_model_parameter_names())
    return model

def fit_model_one_accumulator(df, model_settings, subj_idx):

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
    params['subj_idx'] = subj_idx

    return params


def make_model(sample, model_settings):
    
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
                dx=.005, dt=.01, T_dur=T_dur)
    return model

def fit_model(df, model_settings, subj_idx):

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
    params['subj_idx'] = subj_idx

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
    df = pd.DataFrame({
        'rt': np.concatenate((samp.corr, samp.err)),
        'response': np.concatenate((np.ones(len(samp.corr)), np.zeros(len(samp.err))))
        })
    for value, key in zip(ids, model.required_conditions):
        df[key] = value
    
    # fix:
    df.loc[df['rt']<0, 'response'] = 0
    df.loc[df['response']==0, 'rt'] = np.NaN

    return df

def simulate_data_gonogo(df, params, model_settings, subj_idx, nr_trials=10000):

    # make sample:
    sample = Sample.from_pandas_dataframe(df=df, rt_column_name='rt', correct_column_name='response')

    # make model:
    model = make_model_one_accumulator(sample=sample, model_settings=model_settings)

    # laps rate:
    params['pmixturecoef'] = 0.1
    params['rate'] = 1
    params['noise'] = 1

    # set fitted parameters:
    param_names = model.get_model_parameter_names()
    params_to_set = [Fitted(params.loc[(params['subj_idx']==subj_idx), param]) for param in param_names]
    model.set_model_parameters(params_to_set)
    
    # compute number of trials to generate:
    nr_trials = df.shape[0]
    df.loc[:,'trials'] = 1
    trial_nrs = df.groupby(model.required_conditions).count()['trials']
    trial_nrs = (trial_nrs / sum(trial_nrs) * nr_trials).reset_index()
    trial_nrs['trials'] = trial_nrs['trials'].astype(int)
    
    # resample:
    n_jobs = 24
    res = Parallel(n_jobs=n_jobs)(delayed(sample_from_condition)(model=model, ids=ids, nr_trials=int(d['trials']))
                                    for ids, d in trial_nrs.groupby(model.required_conditions))
    
    # concat:
    df_sim = pd.concat(res)

    # add:
    df_sim['subj_idx'] = subj_idx
    df_sim['hit'] = 0
    df_sim['fa'] = 0
    df_sim['miss'] = 0
    # df_sim['cr'] = 0
    df_sim.loc[(df_sim['rt']>df_sim['start'])&(df_sim['rt']<=(df_sim['start']+df_sim['duration']))&(df_sim['response']==1), 'hit'] = 1
    df_sim.loc[(df_sim['rt']<=df_sim['start'])&(df_sim['response']==1), 'fa'] = 1
    df_sim.loc[(df_sim['hit']==0)&(df_sim['fa']==0), 'miss'] = 1
    df_sim['correct'] = (df_sim['hit']==1)

    return df_sim