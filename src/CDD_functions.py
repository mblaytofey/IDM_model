import pandas as pd
import numpy as np
import glob as glob
import os,sys
import scipy as sp
from scipy import optimize
import math
import matplotlib.pyplot as plt



def fit_delay_discount_model(data_choice_amt_wait, gk_guess = [0.15, 0.5],gk_bounds = ((0,8),(0.0022,7.875)),disp=False):
    # We do start the optimizer off with the guesses above, but those aren't updated like Bayesian priors. 
    # They are simply a starting point in parameter space for the optimizer. Changes here could be an avenue 
    # to explore when seeking to improve performance.
    # [gamma, kappa]
    # gk_guess = [0.15, 0.5]

    # These are the bounds on gamma and kappa. The first tuple corresponds to gamma, the second to kappa.
    # most impulsive person to least impulsive person
    # most patient person >>> lowest kappa
    # least patient person >>> highest kappa
    # beta = 0, flat prob across SV_delta
    # beta = 8 approximates a step function at SV_delta = 0
    # gk_bounds = ((0,8),(1e-8,6.4))
    # alpha = 1
    # These are the inputs of the local_negLL function. They'll be passed through optimize_me()

    inputs = data_choice_amt_wait.T.values.tolist()
    # print(inputs)

    # If seeking to improve performance, could change optimization method, could change maxiter(ations), 
    # or could fiddle with other things. 
    # You might be able to change the distance between steps in the optimzation.
    # results = optimize.minimize(optimize_me,guesses,inputs, bounds = bkbounds,method='L-BFGS-B', 
    #                             tol=1e-5, callback = None, options={'maxiter':10000, 'disp': True})
    results = optimize.minimize(optimize_me, gk_guess, inputs, bounds = gk_bounds,
                                method='L-BFGS-B', options={'disp':disp})
    negLL = results.fun
    gamma = results.x[0]
    kappa = results.x[1]
    
    return negLL, gamma, kappa


def optimize_me(gamma_kappa, inputs):
    choice,value_soon,time_soon,value_delay,time_delay,alpha = inputs
    return function_negLL(choice,value_soon,time_soon,value_delay,time_delay,gamma_kappa,alpha)


def function_negLL(choice,value_soon,time_soon,value_delay,time_delay,gamma_kappa,alpha):

    p_choose_delay = probability_choose_delay(value_soon,time_soon,value_delay,time_delay,gamma_kappa,alpha)[0]
    p_choose_delay = np.array(p_choose_delay)
    choice = np.array(choice)

    # Trap log(0). This will prevent the code from trying to calculate the log of 0 in the next section.
    p_choose_delay[p_choose_delay==0] = 1e-6
    p_choose_delay[p_choose_delay==1] = 1-1e-6
    
    # Log-likelihood
    LL = (choice==1)*np.log(p_choose_delay) + ((choice==0))*np.log(1-p_choose_delay)

    # Sum of -log-likelihood
    negLL = -sum(LL)

    return negLL


def probability_choose_delay(value_soon,time_soon,value_delay,time_delay,gamma_kappa,alpha):
    p_choose_delay = []
    SV_soon = []
    SV_delay = []
    for i,(vs,ts,vd,td,r) in enumerate(zip(value_soon,time_soon,value_delay,time_delay,alpha)):
        # SV_soon for immediate reward
        iSV_soon = SV_discount(vs,ts,gamma_kappa[1],r)
        # SV_delay for larger later 
        iSV_delay = SV_discount(vd,td,gamma_kappa[1],r)

        try: 
            p = 1 / (1 + math.exp(-gamma_kappa[0]*(iSV_delay-iSV_soon)))
            # p = 1 / (1 + math.exp(beta_and_k_array[0]*(SS_SV-LL_SV)))     ## Math.exp does e^(). In other words, if the smaller-sooner SV is higher than the larger-later SV, e^x will be larger, making the denominator larger, making 1/denom closer to zero (low probability of choosing delay). If the LL SV is higher, the e^x will be lower, making 1/denom close to 1 (high probability of choosing delay). If they are the same, e^0=1, 1/(1+1) = 0.5, 50% chance of choosing delay.
        except OverflowError:                                             ## Sometimes the SS_SV is very much higher than the LL_SV. If beta gets too high, the exponent on e will get huge. Math.exp will throw an OverflowError if the numbers get too big. In that case, 1/(1+[something huge]) is essentially zero, so we just set it to 0.
            p = 0
        p_choose_delay.append(p)
        SV_soon.append(iSV_soon)
        SV_delay.append(iSV_delay)
        
    return p_choose_delay,SV_soon,SV_delay


def SV_discount(value,delay,kappa,alpha):
    SV = (value**alpha)/(1+kappa*delay)
    return SV


# Hessian unavailable in this optimization function, but would use results.hess_inv here
#Tester line if you want: print("LL",LL,"AIC",AIC,"BIC",BIC,"R2",r2,"correct",correct)


def get_data(df,cols,alpha_hat=1):
    # select from columns
    data = df[cols]
    # drop rows with NA int them
    data = data.dropna()
    # add alpha column, will change later
    data['alpha']=alpha_hat
    # resp.corr = 0 is lottery, resp.corr = 1 is safe $5
    data['cdd_trial_resp.corr'] = 1.0 - data['cdd_trial_resp.corr']
    percent_impulse = 1.0 - 1.0*data['cdd_trial_resp.corr'].sum()/data['cdd_trial_resp.corr'].shape[0]
    return data,percent_impulse


def count_trial_type(df_col=[],trial_type='task'):
    trial_type_list = df_col.unique()

    if trial_type in trial_type_list:
        # number of instances for practice
        try:
            trial_type_nb = df_col.value_counts()[trial_type]
        except Exception as err:
            print(df_col)
            print('We have an Exception : {}'.format(err))
            sys.exit()
    else:
        trial_type_nb = 0

    return trial_type_nb

# written generically for task so we can use for CDD and CRDM
# This will save two columns for each subject: confidence and SV_delta
# These outputs will be used by Corey Zimba for modeling confidence
def store_SV(fn,df,SV_delta,task='cdd',use_alpha=False,verbose=False):
    # task specific columns
    trial_type_col = '{}_trial_type'.format(task)
    conf_resp = '{}_conf_resp.keys'.format(task)

    practice_nb = count_trial_type(df_col=df[trial_type_col],trial_type='practice')
    task_nb = count_trial_type(df_col=df[trial_type_col],trial_type='task')
    
    if task_nb != len(list(SV_delta)):
        print('Somehow the number of tasks and length of subject values are different')
        raise ValueError
    try:
        df['SV_delta'] = practice_nb*['']+list(SV_delta)
    except ValueError:
        print('We found a ValueError, please inspect spreadsheet and try again')
        sys.exit()
    df_out = df.loc[df[trial_type_col]=='task',[conf_resp,'SV_delta']].reset_index(drop=True)
    df_out = df_out.astype(float)
    df_out['confidence'] = df_out[conf_resp]*df_out['SV_delta']/df_out['SV_delta'].abs()
    df_out.drop(columns=[conf_resp],inplace=True)
    # print(df_out)
    if use_alpha:
        fn = fn.replace('.csv','_SV_hat_alpha.csv')
    else:
        fn = fn.replace('.csv','_SV_hat.csv')
    if verbose:
        print('We will save columns of interest from {} file to : {}'.format(task.upper(),fn))
    df_out.to_csv(fn,index=False)



def main(args):
    print(args)

if __name__ == '__main__':
    main(sys.argv)


