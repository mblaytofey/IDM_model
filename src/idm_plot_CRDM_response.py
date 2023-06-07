import pandas as pd
import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import model_functions as mf
from idm_split_data import make_dir


def columns_there(df):
    cols_check = ['crdm_trial_resp.corr','crdm_lott_top','crdm_lott_bot',
                  'crdm_sure_p','crdm_lott_p','crdm_amb_lev']
    for c in cols_check:
        if c not in list(df):
            print('Moving on to next subject, we could not find column : {}'.format(c))
            return 0
    return 1

def gen_trial_type(df):
    practice = ['practice']*df['crdm_pract_trials.thisTrialN'].count()
    task = ['task']*df['crdm_trials.thisN'].count()
    df['crdm_trial_type'] = practice+task
    return df

def rename_columns(df):
    df = gen_trial_type(df)
    cols = ['sure_amt','sure_p','lott_top','lott_bot','lott_p','amb_lev']
    crdm_cols = ['crdm_{}'.format(c) for c in cols]
    cols_dict = dict([(k,v) for (k,v) in zip(cols,crdm_cols)])
    df.rename(columns=cols_dict,inplace=True)
    return df


# can rewrite in terms of sort, fit, plot, like Corey Z does
def load_estimate_CRDM_save(split_dir='/tmp/',new_subjects=[],task='crdm',verbose=False):
    if verbose:
        print('We are working under /split_dir/ : {}'.format(split_dir))
    crdm_files = mf.get_task_files(split_dir=split_dir,new_subjects=new_subjects,task=task)

    df_cols = ['subject','task','response_rate','percent_lottery','percent_risk','percent_ambiguity',
        'conf_1','conf_2','conf_3','conf_4','negLL','gamma','beta','alpha','at_bound','LL','LL0',
        'AIC','BIC','R2','correct','prob_span','fig_fn']
    df_out = pd.DataFrame(columns=df_cols)

    utility_dir = split_dir.replace('split','utility')
    make_dir(utility_dir)
    batch_name = mf.get_batch_name(split_dir=split_dir)
    df_dir = utility_dir
    df_fn = os.path.join(df_dir,'{}_CRDM_analysis.csv'.format(batch_name))

    # gamma, beta, alpha bounds
    # beta should be [-1.something, 1.something] look at choice set space to determine
    gba_bounds = ((0,8),(-4.167,4.167),(0.125,4.341))
    counter = 0
    for index,fn in enumerate(crdm_files):
        # Load the CDD file and do some checks
        print('Working on CRDM csv file {} of {}:\n{}'.format(index+1,len(crdm_files),fn))
        subject = mf.get_subject(fn,task=task)
        crdm_df = pd.read_csv(fn) #index_col=0 intentionally omitted
        crdm_df,response_rate = mf.drop_non_responses(crdm_df)
        conf_1,conf_2,conf_3,conf_4 = mf.conf_distribution(crdm_df,task=task)        
        if response_rate < 0.05:
            print('**ERROR** Low response rate, cannot model this subjects CRDM data')
            continue

        if not columns_there(crdm_df):
            # hack for columns not being named properly, happened with SDAN data, check again
            print('Checking if renaming columns work')
            crdm_df = rename_columns(crdm_df)
            if not columns_there(crdm_df):
                print('Tried renaming and did not work, check .csv file and try again')
                continue
            elif verbose:
                print('Renaming worked, we will continue as such')
                print(crdm_df)

        cols = ['crdm_trial_resp.corr','crdm_sure_amt','crdm_lott_amt','crdm_sure_p','crdm_lott_p',
            'crdm_amb_lev']
        data,percent_safe = mf.get_data(crdm_df,cols)
        percent_lott = 1.0 - percent_safe
        percent_risk,percent_ambig = mf.percent_risk_ambig(data)
        # Estimate gamma, beta, and alpha
        gba_guess = [0.15, 0.5, 0.6]
        negLL,gamma,beta,alpha = mf.fit_computational_model(data,guess=gba_guess,bounds=gba_bounds,
            disp=False)

        parms_list = [gamma,beta,alpha]
        at_bound = mf.check_to_bound(parms_list,bounds=gba_bounds)
        if verbose:
            print('Percent Risky Choice: {}'.format(percent_risk))
            print("Negative log-likelihood: {}, gamma: {}, beta: {}, alpha: {}".
                  format(negLL, gamma, beta, alpha))

        parms = np.array(parms_list)
        p_choose_reward, SV, fig_fn, choice = mf.plot_save(index,fn,data,parms,task=task,
            ylabel='prob_choose_ambig',xlabel='SV difference (SV_lottery - SV_fixed)',verbose=True)
        mf.store_SV(fn,crdm_df,SV,task=task,use_alpha=False)
        LL,LL0,AIC,BIC,R2,correct = mf.GOF_statistics(negLL,choice,p_choose_reward,nb_parms=3)
        p_range = max(p_choose_reward) - min(p_choose_reward)
        
        row = [subject,task.upper(),response_rate,percent_lott,percent_risk,percent_ambig,
            conf_1,conf_2,conf_3,conf_4,negLL,gamma,beta,alpha,at_bound,LL,LL0,AIC,BIC,R2,
            correct,p_range,fig_fn]
        row_df = pd.DataFrame([row],columns=df_cols)
        df_out = pd.concat([df_out,row_df],ignore_index=True)

        counter += 1

    # Save modeled parameters to modeled results
    mf.save_df_out(df_fn,df_out)

    return counter


def main():
    # if running this script on its own, start here
    # split_dir = '/Users/pizarror/mturk/idm_data/split'
    # one time hack
    SDAN_dir = '/Users/pizarror/mturk/idm_data/batch_output/SDAN'
    # split_dir = '/Users/pizarror/mturk/idm_data/batch_output/bonus2'
    load_estimate_CRDM_save(split_dir=SDAN_dir,verbose=True)


if __name__ == "__main__":
    main()










