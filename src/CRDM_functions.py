import pandas as pd
import numpy as np
import glob as glob
import os,sys
import scipy as sp
from scipy import optimize
import math
import matplotlib.pyplot as plt



def fit_ambiguity_risk_model(data_choice_sure_lott_amb, gba_guess = [1, 0.5, 0.6],
                             gba_bounds = ((0,8),(1e-8,6.4),(0.125,4.341)),disp=False):
    # We do start the optimizer off with the guesses.
    # The guesses are a starting point in parameter space for the optimizer. 

    # These are the bounds on betas. The first tuple corresponds to softmax slope, 
    # the second to beta on ambiguity, the third to value exponent, alpha.
    # most risky person to least risky person
    # most patient person >>> ???
    # least patient person >>> ???
    # beta0 = 0, Prob = 0.5
    # beta0 = 8 goes to infinity, prob=step function
    # bkbounds = ((0,8),(1e-8,6.4),(1e-8,6.4))

    # These are the inputs of the local_negLL function. They'll be passed through optimize_me()
    inputs = data_choice_sure_lott_amb.T.values.tolist()
    # print(inputs)

    # If seeking to improve performance, could change optimization method, could change maxiter(ations), 
    # or could fiddle with other things. 
    # You might be able to change the distance between steps in the optimzation.
    # results = optimize.minimize(optimize_me,guesses,inputs, bounds = bkbounds,method='L-BFGS-B', 
    #                             tol=1e-5, callback = None, options={'maxiter':10000, 'disp': True})
    results = optimize.minimize(optimize_me,gba_guess,inputs,bounds = gba_bounds,
                                method='L-BFGS-B',options={'disp':disp})
    negLL = results.fun
    gamma = results.x[0]
    beta = results.x[1]
    alpha = results.x[2]
    
    return negLL, gamma, beta, alpha


def optimize_me(gamma_beta_alpha, inputs):
    choice,value_fix,value_ambig,p_fix,p_ambig,ambiguity = inputs
    return function_negLL(gamma_beta_alpha,choice,value_fix,value_ambig,p_fix,p_ambig,ambiguity)


def function_negLL(gamma_beta_alpha,choice,value_fix,value_ambig,p_fix,p_ambig,ambiguity):

    p_choose_ambig = probability_choose_ambiguity(value_fix,value_ambig,p_fix,p_ambig,ambiguity,gamma_beta_alpha)[0]
    p_choose_ambig = np.array(p_choose_ambig)
    choice = np.array(choice)

    # Trap log(0). This will prevent the code from trying to calculate the log of 0 in the next section.
    p_choose_ambig[p_choose_ambig==0] = 1e-6
    p_choose_ambig[p_choose_ambig==1] = 1-1e-6
    
    # Log-likelihood
    LL = (choice==1)*np.log(p_choose_ambig) + ((choice==0))*np.log(1-p_choose_ambig)

    # Sum of -log-likelihood
    negLL = -sum(LL)

    return negLL


def probability_choose_ambiguity(value_fix,value_ambig,p_fix,p_ambig,ambiguity,gamma_beta_alpha):
    p_choose_ambig = []
    SV_fix = []
    SV_ambig = []
    ambig_fix = 0
    for i,(vf,va,pf,pa,a) in enumerate(zip(value_fix,value_ambig,p_fix,p_ambig,ambiguity)):
        # subjective value (utility) fixed (sure bet)
        iSV_fix = SV_ambiguity(vf,pf,ambig_fix,gamma_beta_alpha[2],gamma_beta_alpha[1])
        # subjective value (utility) ambiguous (lottery)
        iSV_ambig = SV_ambiguity(va,pa,a,gamma_beta_alpha[2],gamma_beta_alpha[1])

        try: 
            p = 1 / (1 + math.exp(-gamma_beta_alpha[0]*(iSV_ambig - iSV_fix)))
            # p = 1 / (1 + math.exp(beta_and_k_array[0]*(SS_SV-LL_SV)))     ## Math.exp does e^(). In other words, if the smaller-sooner SV is higher than the larger-later SV, e^x will be larger, making the denominator larger, making 1/denom closer to zero (low probability of choosing delay). If the LL SV is higher, the e^x will be lower, making 1/denom close to 1 (high probability of choosing delay). If they are the same, e^0=1, 1/(1+1) = 0.5, 50% chance of choosing delay.
        except OverflowError:                                             ## Sometimes the SS_SV is very much higher than the LL_SV. If beta gets too high, the exponent on e will get huge. Math.exp will throw an OverflowError if the numbers get too big. In that case, 1/(1+[something huge]) is essentially zero, so we just set it to 0.
            p = 0
        p_choose_ambig.append(p)
        SV_fix.append(iSV_fix)
        SV_ambig.append(iSV_ambig)
        
    return p_choose_ambig,SV_fix,SV_ambig


def SV_ambiguity(value,p_win,ambiguity,alpha,beta):
    # subjective value, SV, different when positive and negative
    if value>0:
        SV = (p_win - beta*ambiguity/2) * (value**alpha)
    else:
        SV = (p_win - beta*ambiguity/2) *(-1.0)*(abs(value)**alpha)
    return SV


def get_task_files(split_dir='/tmp/',task='crdm'):
    task_files = glob.glob(os.path.join(split_dir,'*/*/*_{}*.csv'.format(task)))
    task_files = [f for f in task_files if 'SV_hat.csv' not in f]
    if not task_files:
        print('\n\n***ERROR***\nThe path to split_dir did not have any .csv files for analysis.\n\n')
        print('Check input path again and rerun script : {}'.format(split_dir))
        sys.exit()        
    return task_files


def drop_non_responses(df):
    keys_cols = [c for c in list(df) if 'trial_resp.keys' in c]
    if not keys_cols:
        print('We found no column with a trial_resp.keys in the name, check .csv file before continuing. These are the columns names:')
        print(list(df))
        sys.exit()
    if len(keys_cols)==2:
        # this should be the most common number of keys_cols
        df['responded'] = df[keys_cols[0]].notna() | df[keys_cols[1]].notna()
    elif len(keys_cols)==1:
        print('**WARNING** Only found one trial_resp.keys : {}'.format(keys_cols))
        print('We will continue with what we have but check to make sure this is what you want')
        df['responded'] = df[keys_cols[0]].notna()
    elif len(keys_cols)==3:
        print('**WARNING** Found three trial_resp.keys : {}'.format(keys_cols))
        print('We will continue with what we have but check to make sure this is what you want')
        df['responded'] = df[keys_cols[0]].notna() | df[keys_cols[1]].notna() | df[keys_cols[2]].notna()
    else:
        print('**ERROR** Found too many trial_resp.keys : {}'.format(keys_cols))
        print('Check your file and try again')
        sys.exit()

    response_rate=1.0
    if not df['responded'].all():
        non_responses_nb = df['responded'].value_counts()[False]
        response_rate -= float(non_responses_nb)/df['responded'].shape[0]
        print('\n**WARNING** We dropped {0} of {1} non responses, resulting in response_rate of : {2:0.3f}\n'.format(non_responses_nb,df['responded'].shape[0],response_rate))
        df = df.loc[df['responded'],:].reset_index(drop=True)
    return df,response_rate


def get_subject(fn,task='crdm'):
    return os.path.basename(fn).replace('_{}.csv'.format(task),'')



def get_fig_fn(fn,use_alpha=False):
    split_dir = os.path.dirname(os.path.dirname(os.path.dirname(fn)))
    if use_alpha:
        fig_fn = fn.replace(split_dir,'').replace('.csv','_model_fit_alpha.png')[1:]
    else:
        fig_fn = fn.replace(split_dir,'').replace('.csv','_model_fit.png')[1:]
    return split_dir,fig_fn


def get_data(df,cols):
    # combining top and bottom values into amount column
    df['crdm_lott_amt'] = df['crdm_lott_top'] + df['crdm_lott_bot']
    # select from columns
    data = df[cols]
    # drop rows with NA int them
    data = data.dropna()
    # resp.corr = 0 is lottery, resp.corr = 1 is safe $5
    data['crdm_trial_resp.corr'] = 1.0 - data['crdm_trial_resp.corr']
    percent_risk = 1.0 - 1.0*data['crdm_trial_resp.corr'].sum()/data['crdm_trial_resp.corr'].shape[0]
    return data,percent_risk


def GOF_statistics(negLL,choice,p_choice,nb_parms=2):
    # Unrestricted log-likelihood
    # LL = (choice==1)*np.log(p_ambig) + ((choice==0))*np.log(1-p_ambig)
    LL = -negLL

    # Restricted log-likelihood, baseline comparison
    LL0 = np.sum(sum(choice)*np.log(0.5) + (len(choice)-sum(choice))*np.log(0.5)) # + np.finfo(np.float32).eps
    
    # Akaike Information Criterion
    AIC = -2*LL + 2*nb_parms  #CHANGE TO len(results.x) IF USING A DIFFERENT MODEL (parameters != 2)

    # Bayesian information criterion
    BIC = -2*LL + 2*math.log(len(p_choice))  #len(results.x)

    #R squared
    r2 = 1 - LL/LL0
    # r2 = 1 - (math.exp(LL-LL0))**(-2/len(choice))

    #Percent accuracy
    p = np.array(p_choice) # gets an array of probabilities of choosing the LL choice
    correct = sum((p>=0.5)==choice)/len(p_choice)                                          # LL is 1 in choices, so when the parray is > 0.5 and choices==1, the model has correctly predicted a choice.

    return LL,LL0,AIC,BIC,r2,correct

# Hessian unavailable in this optimization function, but would use results.hess_inv here
#Tester line if you want: print("LL",LL,"AIC",AIC,"BIC",BIC,"R2",r2,"correct",correct)

def main(args):
    print(args)

if __name__ == '__main__':
    main(sys.argv)

