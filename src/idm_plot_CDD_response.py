import pandas as pd
import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import model_functions as mf
from idm_split_data import make_dir


def columns_there(df):
    cols_check = ['cdd_trial_resp.corr','cdd_immed_amt','cdd_immed_wait','cdd_delay_amt',
                  'cdd_delay_wait','cdd_conf_resp.keys']
    for c in cols_check:
        if c not in list(df):
            print('Moving on to next subject, we could not find column : {}'.format(c))
            return 0
    return 1


def check_to_bound(gamma,kappa,gk_bounds= ((0,8),(1e-8,6.4))):
    at_bound = 0
    if gamma in gk_bounds[0]:
        at_bound = 1
    elif kappa  in gk_bounds[1]:
        at_bound = 1
    return at_bound


def get_alpha_hat(model_dir='/tmp/',batch_name='batch',subject='person1'):
    CRDM_fn = os.path.join(model_dir,'{}_CRDM_analysis.csv'.format(batch_name))
    CRDM_df = pd.read_csv(CRDM_fn,index_col=0)
    # using .loc function to find the alpha value, but still need get item
    try:
        alpha_hat = CRDM_df.loc[CRDM_df['subject']==subject,'alpha'].item()
    except ValueError:
        print('We have a ValueError, will just set alpha to 1.0')
        alpha_hat = 1.0
    return alpha_hat

# can rewrite in terms of sort, fit, plot, like Corey Z does
def load_estimate_CDD_save(split_dir='/tmp/',new_subjects=[],task='cdd',use_alpha=False,verbose=False):
    if verbose:
        print('We are working under /split_dir/ : {}'.format(split_dir))
    cdd_files = mf.get_task_files(split_dir=split_dir,new_subjects=new_subjects,task=task)
    # cdd_files = glob.glob(os.path.join(split_dir,'*/*/*_cdd.csv'))

    df_cols = ['subject','task','response_rate','percent_impulse','conf_1','conf_2','conf_3','conf_4',
        'negLL','gamma','kappa','alpha','at_bound','LL','LL0','AIC','BIC','R2','correct','prob_span',
        'fig_fn']
    df_out = pd.DataFrame(columns=df_cols)

    utility_dir = split_dir.replace('split','utility')
    make_dir(utility_dir)
    batch_name = mf.get_batch_name(split_dir=split_dir)
    df_dir = utility_dir
    df_fn = os.path.join(df_dir,'{}_CDD_analysis.csv'.format(batch_name))
    if use_alpha:
        df_fn = df_fn.replace('.csv','_alpha.csv')

    gk_bounds = ((0,8),(0.0022,7.875))
    for index,fn in enumerate(cdd_files):
        # Load the CDD file and do some checks
        print('Working on CDD csv file {} of {}:\n{}'.format(index+1,len(cdd_files),fn))
        subject = mf.get_subject(fn,task=task)
        cdd_df = pd.read_csv(fn) #index_col=0 intentionally avoided
        cdd_df,response_rate = mf.drop_non_responses(cdd_df)
        conf_1,conf_2,conf_3,conf_4 = mf.conf_distribution(cdd_df,task=task)
        if response_rate < 0.05:
            print('**ERROR** Low response rate, cannot model this subjects CDD data')
            continue

        if not columns_there(cdd_df):
            continue
        
        # default value if not using alpha for modeling
        alpha_hat=1
        if use_alpha:
            alpha_hat = get_alpha_hat(model_dir=df_dir,batch_name=batch_name,subject=subject)
        cols = ['cdd_trial_resp.corr','cdd_immed_amt','cdd_delay_amt','cdd_immed_wait','cdd_delay_wait','alpha']
        data, percent_impulse = mf.get_data(cdd_df,cols,alpha_hat=alpha_hat)
        # Estimate gamma and kappa with or without alpha
        gk_guess = [0.15, 0.5]
        negLL,gamma,kappa = mf.fit_computational_model(data,guess=gk_guess,bounds=gk_bounds,disp=False)

        parms_list = [gamma,kappa]
        at_bound = mf.check_to_bound(parms_list,bounds=gk_bounds)
        if verbose:
            print('From CRDM we estimated the following alpha value : {}'.format(alpha_hat))
            print('Percent Impulse Choice: {}'.format(percent_impulse))
            print("Negative log-likelihood: {}, gamma: {}, kappa: {}".
                  format(negLL, gamma, kappa))

        parms = np.array(parms_list)
        p_choose_reward, SV, fig_fn, choice = mf.plot_save(index,fn,data,parms,task=task,
            ylabel='prob_choose_delay',xlabel='SV difference (SV_delay - SV_immediate)',
            use_alpha=use_alpha,verbose=True)
        mf.store_SV(fn,cdd_df,SV_delta=SV,task=task,use_alpha=use_alpha)
        LL,LL0,AIC,BIC,R2,correct = mf.GOF_statistics(negLL,choice,p_choose_reward,nb_parms=2)
        p_range = max(p_choose_reward) - min(p_choose_reward)
        
        row = [subject,task.upper(),response_rate,percent_impulse,conf_1,conf_2,conf_3,conf_4,
            negLL,gamma,kappa,alpha_hat,at_bound,LL,LL0,AIC,BIC,R2,correct,p_range,fig_fn]
        row_df = pd.DataFrame([row],columns=df_cols)
        df_out = pd.concat([df_out,row_df],ignore_index=True)

    # Save modeled parameters to modeled results
    mf.save_df_out(df_fn,df_out)


def main():
    # if running this script on its own, start here
    # split_dir = '/Users/pizarror/mturk/idm_data/split'
    split_dir = mf.get_split_dir()
    # split_dir = '/Users/pizarror/mturk/idm_data/batch_output/raw/'
    load_estimate_CDD_save(split_dir)



if __name__ == "__main__":
    main()










