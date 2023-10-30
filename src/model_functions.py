import pandas as pd
import numpy as np
import glob as glob
import os,sys
# import scipy as sp
from scipy.stats import bernoulli
from scipy import optimize
import math
import matplotlib.pyplot as plt


'''
### Pre model-fitting functions, to prepare data and files ###

'''

# get split_dir for each project
def get_split_dir():
	split_dir = input('What is the split directory? Where are the task files located?\n')
	return split_dir

# search for task files under split_dir and return the list of files, throw error if nothing found
def get_task_files(split_dir='/tmp/',new_subjects=[],task='crdm'):
    task_files = glob.glob(os.path.join(split_dir,'*/*/*_{}*.csv'.format(task)))
    task_files = [f for f in task_files if 'SV_hat' not in f]
    if new_subjects:
        task_files = [f for f in task_files if get_subject(f,task=task) in new_subjects]
    if not task_files:
        print('\n\n***ERROR***\nThe path to split_dir did not have any .csv files for analysis.\n\n')
        print('Check input path again and rerun script : {}'.format(split_dir))
        sys.exit()        
    return sorted(task_files)

def get_batch_name(split_dir='/tmp/'):
    batch_name = os.path.basename(split_dir)
    # check empty string
    if not batch_name:
        batch_name = os.path.basename(os.path.dirname(split_dir))
        # check empty string a second time
        if not batch_name:
            batch_name = 'batchity_batch'
            print('**WARNING** Could not determine batch name, will use {}, please check your input dir : {}'.format(batch_name,split_dir))
    return batch_name

# simple function to get the subject from the filename for appending to analysis spreadsheet and use as title on plots
def get_subject(fn,task='crdm'):
    if not task:
        return os.path.basename(fn).replace('{}.csv'.format(task),'')
    else:
        return os.path.basename(fn).replace('_{}.csv'.format(task),'')


# confidence responce column has been a moving target. this will help centralize any future fixes if necessary
def get_confresp(df,task='crdm'):
    if '{}_confkey'.format(task) in list(df):
        return '{}_confkey'.format(task)
    elif '{}_conf_resp.keys'.format(task) in list(df):
        return '{}_conf_resp.keys'.format(task)
    else:
        print('Whoops could not find confidence response colums: {}'.format(list(df)))
        print('**EXITING NOW**')
        sys.exit()

def get_choicecol(df,task='crdm'):
    if '{}_choice'.format(task) in list(df):
        return '{}_choice'.format(task)
    else:
        print('Could not find choice column, possible older version of psychopy : {}'.format(list(df)))
        sys.exit()


# split dataframe by gains/losses
def get_by_domain(df,domain='gain',task='crdm',verbose='False'):
    if verbose:
        print('Working on this domain: {}'.format(domain))
    # select by domain: gain/loss
    domain_col = '{}_domain'.format(task)
    df = df.loc[df[domain_col]==domain]
    return df

# prefer to use crdm_choice and cdd_choice but catch in case it does not exist
# simple function to remap the responses if necessary and store into crdm_choice
def remap_response(df,task='crdm'):
    if '{}_choice'.format(task) in list(df):
        return df
    elif task=='cpdm':
        resp_key_col = '{}_trial_resp.keys'.format(task)
        # colum saved as resp.keys = ['q','p','a','l'] :: [-2,2,-1,1]
        # 1/2 distinguish low/high confidence
        # +/- distinguish left and right orientation
        choice_dict = {'q':-2,'p':2,'a':-1,'l':1}
        # create task_choice
        task_choice = [c if len(c)==0 else choice_dict[c] for c in df[resp_key_col].values]
        df['{}_choice'.format(task)] = task_choice
    else:
        # create task_choice
        resp_corr_col = '{}_trial_resp.corr'.format(task)
        resp_key_col = '{}_trial_resp.keys'.format(task)
        # colum saved as resp.corr = 0 is reward, resp.corr = 1 is null 
        # want to use as resp.corr = 1 is reward, resp.corr = 0 is null
        task_choice = [k if math.isnan(k) else 1-c for (c,k) in zip(df[resp_corr_col].values,df[resp_key_col].values)]
        df['{}_choice'.format(task)] = task_choice
    return df

# simple fucntion to drop practice trials. They are not used
def drop_pract(df,task='crdm'):
    trial_type_col = '{}_trial_type'.format(task)
    df = df.loc[df[trial_type_col]=='task']
    cols = [c for c in list(df) if 'pract' not in c]
    df = df[cols]
    return df

# Function for dropping blank responses found in either the task or the confidence measure.
# We cannot use data that is blank, so we remove and count the number of blanks found and report it
## this need to have the flexibility to work for both when we have Nan or None 
## these two are the options encountered so far
def drop_non_responses(df,task='crdm',conf_drop=True,verbose=False):
    # original length of df before dropping rows
    df_len = df.shape[0]
    # get the relevant column names
    choice_col = get_choicecol(df,task=task)
    conf_resp = get_confresp(df,task=task)

    df,nan_nb = drop_by_nan(df,df_len,choice_col,conf_resp,conf_drop=conf_drop,verbose=verbose)

    if conf_drop:
        df,none_nb = drop_by_str(df,col=conf_resp,match_str='None')
    else:
        df,none_nb = drop_by_str(df,col=choice_col,match_str='None')

    # Compute response_rate based on non_responses_nb and None_drops
    response_rate = 1.0 - float(nan_nb+none_nb)/df_len

    if verbose and (response_rate < 1.0):
        print('The {0} drop(s) resulted in response_rate : {1}\n'.format(nan_nb+none_nb,response_rate))

    return df,response_rate

def drop_by_nan(df,df_len,choice_col,conf_resp,conf_drop=True,verbose=False):
    # initialized to avoid errors
    non_responses_nb = 0
    if conf_drop:
        # dropping Nan from response and confidence 
        df['responded'] = df[conf_resp].notna()
    else:
        # this should be the most common number of keys_cols
        df['responded'] = df[choice_col].notna()

    if not df['responded'].all():
        non_responses_nb = df['responded'].value_counts()[False]
        if verbose:
            print('\n**WARNING** We dropped {0} of {1} CHOICE responses that were Nan'.format(non_responses_nb,df_len))
        df = df.loc[df['responded'],:].reset_index(drop=True)
    return df,non_responses_nb

# written for SDAN data, when None started appearing instead of empty or Nan, can match any string, default to 'None'
# The None shows up as a Nan on my laptop but 'None' in other computers
# crdm_confkey
def drop_by_str(df,col='crdm_choice',match_str='None'):
    drops=0
    if df[ col ].dtype == 'float64':
        return df,drops

    df1_len = df.shape[0]
    try:
        df = df.loc[ df[ col ].str.contains( match_str )==False ].reset_index(drop=True)
    except AttributeError:
        print(df[col])
        print('Something up with col : {}'.format(col))
        sys.exit()
    df2_len = df.shape[0]
    drops = df1_len-df2_len
    if drops>0:
        print('**WARNING** We dropped {} rows from column {} containing >>>{}<<<\n'.format(drops,col,match_str))
    return df,drops

# After dropping the blank rows, we can compute confidence distribution
def conf_distribution(df,task='crdm'):
    trial_type_col = '{}_trial_type'.format(task)
    df = df.loc[df[trial_type_col]=='task']
    conf_resp = get_confresp(df,task=task) #'{}_conf_resp.keys'.format(task)
    counts = df[conf_resp].value_counts()
    # initialize at 0
    count_list = [0]*4
    for i in counts.index:
        if type(i) is str:
            if i in ['1','2','3','4']:
                pass
            else:
                print('**WARNING** We found a string >>>{}<<< in the conf_resp column. We will skip for now'.format(i))
                continue
        count_list[int(i)-1]=counts[i]
    return tuple(count_list)

# We select the columns of interest so we can model with the computational models
def get_data(df,cols,alpha_hat=1.0,domain='gain',task='crdm'):
    task = get_task(df)
    if task == 'crdm':
        # combining top and bottom values into amount column
        df['crdm_lott_amt'] = df['crdm_lott_top'] + df['crdm_lott_bot']
        # convert percentage to probabilities
        df['crdm_sure_p'] = df['crdm_sure_p'] / 100.0
        df['crdm_lott_p'] = df['crdm_lott_p'] / 100.0
        df['crdm_amb_lev'] = df['crdm_amb_lev'] / 100.0
    elif task == 'cdd':
        # add alpha column, will change later
        df['alpha']=alpha_hat

    # select from columns
    data = df[cols]
    # drop rows with NA int them
    data = data.dropna()

    choice_col = get_choicecol(df,task=task)
    # resp_corr_col = '{}_trial_resp.corr'.format(task)
    # crdm: percent_safe, cdd: percent_impulse
    percent_null = 1.0 - 1.0*data[choice_col].sum()/data[choice_col].shape[0]

    return data,percent_null

def percent_risk_ambig(df,task='crdm'):
    # resp_corr_col = next(c for c in list(df) if 'trial_resp.corr' in c)
    choice_col = get_choicecol(df,task=task)
    # resp_corr_col = '{}_trial_resp.corr'.format(task)
    amb_lev_col = next(c for c in list(df) if 'crdm_amb_lev' in c)

    df_risk = df.loc[df[amb_lev_col]==0]
    df_ambig = df.loc[df[amb_lev_col]>0]

    percent_risk = 1.0*df_risk[choice_col].sum()/df_risk[choice_col].shape[0]
    percent_ambig = 1.0*df_ambig[choice_col].sum()/df_ambig[choice_col].shape[0]
    
    return percent_risk,percent_ambig


'''
### Model fitting functions ###

'''

def fit_computational_model(data, guess=[1,0.5,0.6],bounds=((0,8),(1e-8,6.4),(0.125,4.341)),disp=False):
    # data : data_choice_sure_lott_amb for CRDM, 
    #        data_choice_amt_wait for CDD

    #### fit model with the minimize function ####
    # for improvement try methods=['BFGS','SLSQP'] 
    # try initializing with 1000 different values
    results = optimize.minimize(function_negLL,guess,args=data,bounds=bounds,method='L-BFGS-B',options={'disp':disp})
    # results = optimize.minimize(function_negLL,guess,args=data,bounds=bounds,method='L-BFGS-B',options={'disp':disp})
    
    # number of parameters
    nb_parms = len(guess)
    # list of results to turn to tuple
    fit_results = [results.fun] + [results.x[i] for i in range(nb_parms)]
    
    return tuple(fit_results)

def get_task(data):
    cols = sorted(list(data))
    # choice_col = next(c for c in cols if ('choice' in c) and ('bonus' not in c))
    # resp_corr_col = next(c for c in cols if 'trial_resp.corr' in c)
    # let's check choice column : trial_resp.corr
    if any('crdm' in c for c in cols):
        return 'crdm'
    elif any('cdd' in c for c in cols):
        return 'cdd'
    else:
        print('We could not find task name from colums : {}'.format(cols))
        sys.exit()

def function_negLL(parms,data):
    # parms : gamma_beta_alpha for CRDM, gamma_kappa_alpha for CDD
    # args = inputs
    # inputs: choice,value_null,value_reward,p_null,p_reward,ambiguity for CRDM
    #         choice,value_soon,value_delay,time_soon,time_delay,alpha for CDD

    task = get_task(data)
    if task == 'crdm':
        cols = ['crdm_choice','crdm_sure_amt','crdm_lott_amt','crdm_sure_p','crdm_lott_p','crdm_amb_lev']
        # choice,value_null,value_reward,p_null,p_reward,ambiguity = data.T.values.tolist()
        choice,value_null,value_reward,p_null,p_reward,ambiguity = extract_data(data,cols=cols)
        p_choose_reward = probability_choice(parms,value_null,value_reward,p_null=p_null,p_reward=p_reward,ambiguity=ambiguity,task=task)[0]
    elif task == 'cdd':
        cols = ['cdd_choice','cdd_immed_amt','cdd_delay_amt','cdd_immed_wait','cdd_delay_wait','alpha']
        # choice,value_null,value_reward,time_null,time_reward,alpha = data.T.values.tolist()
        choice,value_null,value_reward,time_null,time_reward,alpha = extract_data(data,cols=cols)
        p_choose_reward = probability_choice(parms,value_null,value_reward,time_null=time_null,time_reward=time_reward,alpha=alpha,task=task)[0]

    p_choose_reward = np.array(p_choose_reward)
    choice = np.array(choice)

    # Trap log(0). This will prevent the code from trying to calculate the log of 0 in the next section.
    p_choose_reward[p_choose_reward==0] = 1e-6
    p_choose_reward[p_choose_reward==1] = 1-1e-6
    
    # Log-likelihood
    # LL = (choice==1)*np.log(p_choose_reward) + ((choice==0))*np.log(1-p_choose_reward)
    LL = bernoulli.logpmf(choice, p_choose_reward)
    # Sum of -log-likelihood
    negLL = -sum(LL)

    return negLL


def extract_data(data,cols=[]):
    # cols = ['crdm_choice','crdm_sure_amt','crdm_lott_amt','crdm_sure_p','crdm_lott_p','crdm_amb_lev']
    data_cols = ()
    for c in cols:
        data_cols += (data[c].T.values.tolist(),)
    return data_cols

def prob_softmax(SV1,SV0,gamma=0.5):
    # compute probability using softmax function, return 0 if OverlowError is thrown
    try: 
        p = 1 / (1 + math.exp(-gamma*(SV1 - SV0)))
    except OverflowError:
        p = 0
    return p

def append_prob_SV(p_choose_reward,SV_null,SV_reward,parms,SV1,SV0):
    # compute prob based on SV values
    p = prob_softmax(SV1,SV0,gamma=parms[0])
    # append to list
    p_choose_reward.append(p)
    SV_null.append(SV0)
    SV_reward.append(SV1)
    return p_choose_reward,SV_null,SV_reward

def probability_choice(parms,value_null,value_reward,p_null=[1.0],p_reward=[0.5],ambiguity=[0.0],time_null=[0],time_reward=[30],alpha=[1.0],ambig_null=0,task='crdm'):
    p_choose_reward = []
    SV_null = []
    SV_reward = []
    if task=='crdm':
        for vn,vr,pn,pr,a in zip(value_null,value_reward,p_null,p_reward,ambiguity):
            # subjective value (utility) null, reward, corresponding probability choice
            iSV_null = SV_ambiguity(vn,pn,ambig_null,alpha=parms[2],beta=parms[1])
            iSV_reward = SV_ambiguity(vr,pr,a,alpha=parms[2],beta=parms[1])
            p_choose_reward,SV_null,SV_reward = append_prob_SV(p_choose_reward,SV_null,SV_reward,parms,iSV_reward,iSV_null)
    elif task=='cdd':
        for vn,vr,tn,tr,a in zip(value_null,value_reward,time_null,time_reward,alpha):
            # subjective value (utility) null, reward, corresponding probability choice
            iSV_null = SV_discount(vn,tn,kappa=parms[1],alpha=a)
            iSV_reward = SV_discount(vr,tr,kappa=parms[1],alpha=a)
            p_choose_reward,SV_null,SV_reward = append_prob_SV(p_choose_reward,SV_null,SV_reward,parms,iSV_reward,iSV_null)

    return p_choose_reward,SV_null,SV_reward

def SV_ambiguity(value,p_win,ambiguity,alpha=1.0,beta=0.5):
    # subjective value, SV, different when positive and negative
    if value>0:
        SV = (p_win - beta*ambiguity/2) * (value**alpha)
    else:
        SV = (p_win - beta*ambiguity/2) *(-1.0)*(abs(value)**alpha)
    return SV

def SV_discount(value,delay,kappa=0.005,alpha=1.0):
    SV = (value**alpha)/(1+kappa*delay)
    return SV




'''
### Post model-fitting functions ###

'''


# after model fit, check if parameter estimates are located at the bound
def check_to_bound(parms,bounds= ((0,8),(1e-8,6.4),(1e-8,6.4))):
    at_bound = 0
    for i,p in enumerate(parms):
        if p in bounds[i]:
            at_bound = 1
            return at_bound
    return at_bound

# Function to plot the model fit and the choice data. We plot probability of choice as a function of subjective value
def plot_save(index,fn,data,parms,domain='',task='crdm',ylabel='prob_choose_ambig',xlabel='SV difference',use_alpha=False,verbose=False):
    # CDD title, add domain for CRDM
    title = get_subject(fn,task=task)

    # extract probability and SV by plugging in estimates parameters into probability choice along with lists of values (choice_set_space)
    if task=='crdm':
        title = '{} {}'.format(title,domain)
        choice,value_null,value_reward,p_null,p_reward,ambiguity = data.T.values.tolist()
        p_choose_reward,SV_null,SV_reward = probability_choice(parms,value_null,value_reward,p_null=p_null,p_reward=p_reward,ambiguity=ambiguity,task=task)
    elif task=='cdd':
        choice,value_null,value_reward,time_null,time_reward,alpha = data.T.values.tolist()
        p_choose_reward,SV_null,SV_reward = probability_choice(parms,value_null,value_reward,time_null=time_null,time_reward=time_reward,alpha=alpha,task=task)
    else:
        print('could not estimate values as value for set was : {}'.format(task))
        sys.exit()

    SV_delta = [rew-null for (rew,null) in zip(SV_reward,SV_null)]
    # for saving
    SV = SV_delta
    # sorted for plotting
    SV_delta, p_choose_reward, choice = zip(*sorted(zip(SV_delta, p_choose_reward, choice)))

    utility_dir,fig_fn = get_fig_fn(fn,domain=domain,use_alpha=use_alpha,task=task)
    plt = plot_fit(index,parms,SV_delta,p_choose_reward,choice=choice,ylabel=ylabel,xlabel=xlabel,title='')

    if verbose:
        plt.title(title,fontsize=15)
        print('Saving to : /utility_dir/ {}'.format(utility_dir))
    plt.savefig(os.path.join(utility_dir,fig_fn),format='eps')
    plt.close(index)
    return p_choose_reward, SV, fig_fn, choice

# function to plot the fit, can be used independently
def plot_fit(index,parms,SV_delta,p_choose_reward,choice=[],ylabel='prob_choose_ambig',xlabel='SV difference',title=''):
    plt.figure(index)

    prob_fit,SV_fit = fitted_model(parms,SV_delta)
    plt.plot(SV_fit,prob_fit,'b-',linewidth=0.5)

    if choice:
        plt.plot(SV_delta,choice,'r*-',linewidth=0.5)
    else:
        plt.plot(SV_delta,p_choose_reward,'r*-',linewidth=0.5)
    plt.plot([min(SV_delta),max(SV_delta)],[0.5,0.5],'k--',linewidth=0.5)
    plt.plot([0,0],[0.0,1.0],'k--',linewidth=0.5)

    plt.ylabel(ylabel,fontsize=12)
    plt.xlabel(xlabel,fontsize=12)
    if title:
        plt.title(title,fontsize=15)
    return plt

def fitted_model(parms,SV_delta):
    gamma = parms[0]
    SV_fit = np.linspace(min(SV_delta),max(SV_delta),300)
    prob_fit = [prob_softmax(sv,0,gamma=gamma) for sv in SV_fit]
    return prob_fit,SV_fit

def make_dir(this_dir,verbose=False):
    if not os.path.exists(this_dir):
        if verbose:
            print('Creating: {}'.format(this_dir))
        os.makedirs(this_dir)

# Function to produce a filename for the figure, we use the task spreadsheet and change it to a png file
def get_fig_fn(fn,task='cdd',domain='gain',use_alpha=False):
    fig_dir = os.path.dirname(fn).replace('split','utility')
    if use_alpha:
        fig_dir = fig_dir.replace(task,'{}_nlh'.format(task))
    make_dir(fig_dir)
    split_dir = os.path.dirname(os.path.dirname(os.path.dirname(fn)))
    utility_dir = os.path.dirname(os.path.dirname(fig_dir))
    fig_fn = fn.replace(split_dir,'').replace('.csv','_model_fit.eps')[1:]
    if len(domain)>0:
        fig_fn = fn.replace(split_dir,'').replace('.csv','_{}_model_fit.eps'.format(domain))[1:]
    if use_alpha:
        fig_fn = fig_fn.replace('/cdd/','/cdd_nlh/')
        fig_fn = fig_fn.replace('_model_fit.eps','_nlh_model_fit.eps')
        # fig_fn = fig_fn.replace('_model_fit.eps','_model_fit_alpha.eps')
    return utility_dir,fig_fn

# function to count the number of trial types, some data was giving a problem and length was not matching, this is fail safe
# called by store_SV()
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
def store_SV(fn,df,SV_delta=[],domain='',task='cdd',conf_drop=False,use_alpha=False,verbose=False):
    # task specific columns
    trial_type_col = '{}_trial_type'.format(task)
    conf_resp = get_confresp(df,task=task)
    choice_col = get_choicecol(df,task=task)

    practice_nb = count_trial_type(df_col=df[trial_type_col],trial_type='practice')
    task_nb = count_trial_type(df_col=df[trial_type_col],trial_type='task')

    if task_nb != len(list(SV_delta)):
        print('Number of task trials is : {}'.format(task_nb))
        print('Number of entries in SV_delta is : {}'.format(len(list(SV_delta))))
        print('Somehow the number of tasks and length of subject values are different')
        raise ValueError
    try:
        df['SV_delta'] = practice_nb*['']+list(SV_delta)
        df['ambig_trial'] = 0
        if task=='crdm':
            df['ambig_trial'] = df['crdm_amb_lev'] > 0
    except ValueError:
        print('We found a ValueError, please inspect spreadsheet and try again')
        sys.exit()
    df_out = df.loc[df[trial_type_col]=='task',[conf_resp,choice_col,'SV_delta','ambig_trial']].reset_index(drop=True)

    if not conf_drop:
        # update df_out for missing confidence responses
        df_out['responded'] = df_out[conf_resp].notna()
        if not df_out['responded'].all():
            non_responses_nb = df_out['responded'].value_counts()[False]
            if verbose:
                print('\n**WARNING** We dropped {0} of {1} CONFIDENCE responses that were left blank, not stored in SV_hat'.format(non_responses_nb,df_out.shape[0]))
            df_out = df_out.loc[df_out['responded'],:].reset_index(drop=True)
        df_out = drop_by_str(df_out,col=conf_resp,match_str='None')[0]

    df_out = df_out.astype(float)
    df_out['valence'] = 2.0*df_out[choice_col] - 1.0
    df_out['confidence'] = df_out[conf_resp]*df_out['valence']

    df_out = df_out.loc[:,['SV_delta','ambig_trial','confidence']]
    fn = fn.replace('split','utility').replace('.csv','_SV_hat.csv')
    if len(domain)>0:
        fn = fn.replace('split','utility').replace('.csv','_{}_SV_hat.csv'.format(domain))
    if use_alpha:
        fn_dir = os.path.dirname(fn).replace(task,'{}_nlh'.format(task))
        make_dir(fn_dir)
        fn = os.path.join(fn_dir,os.path.basename(fn).replace('_SV_hat.csv','_nlh_SV_hat.csv'))
        # fn = fn.replace('_SV_hat.csv','_SV_hat_alpha.csv')
    if verbose:
        print('We will save columns of interest from {} file to : {}'.format(task.upper(),fn))
    df_out.to_csv(fn,index=False)

# summary statistics of the goodness of fit of the computational models
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
    R2 = 1 - LL/LL0

    #Percent accuracy
    p = np.array(p_choice)
    correct = sum((p>=0.5)==choice)/len(p_choice)                                          

    return LL,LL0,AIC,BIC,R2,correct

# function to save current analysis 
# if path is there, then append to previous saved analysis
def save_df_out(fn,df1):
    if os.path.exists(fn):
        df0 = pd.read_csv(fn,index_col=0)
        df0 = df0.loc[~df0['subject'].isin(df1['subject'])]
        frames = [df0, df1]
        df1 = pd.concat(frames,ignore_index=True)
    print('Saving analysis to : {}'.format(fn))
    df1.to_csv(fn)
    return



def main(args):
    print(args)

if __name__ == '__main__':
    main(sys.argv)

