import pandas as pd
import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import model_functions as mf
from idm_split_data import make_dir

def columns_there(df):
    cols_check = ['crdm_choice','crdm_lott_top','crdm_lott_bot',
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


def estimate_CRDM_by_domain(crdm_df,fn,index,subject='joe_shmoe',df_cols=[],
                            gba_bounds = ((0,8),(-4.167,4.167),(0.125,4.341)),
                            domain='gain',task='crdm',conf_drop=True,verbose=False):
    
    # crdm_df = mf.remap_response(crdm_df,task=task)
    crdm_df = mf.drop_pract(crdm_df,task=task)
    crdm_df,response_rate = mf.drop_non_responses(crdm_df,task=task,conf_drop=conf_drop,verbose=verbose)
    conf_1,conf_2,conf_3,conf_4 = mf.conf_distribution(crdm_df,task=task)
    if response_rate < 0.05:
        print('**ERROR** Low response rate, cannot model this subjects CRDM data')
        return

    if not columns_there(crdm_df):
        # hack for columns not being named properly, happened with SDAN data, check again
        print('Checking if renaming columns work')
        crdm_df = rename_columns(crdm_df)
        if not columns_there(crdm_df):
            print('Tried renaming and did not work, check .csv file and try again')
            return
        elif verbose:
            print('Renaming worked, we will continue as such')
            print(crdm_df)

    cols = ['crdm_choice','crdm_sure_amt','crdm_lott_amt','crdm_sure_p','crdm_lott_p','crdm_amb_lev']

    data,percent_safe = mf.get_data(crdm_df,cols,task=task)
    percent_lott = 1.0 - percent_safe
    percent_risk,percent_ambig = mf.percent_risk_ambig(data,task=task)
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
    p_choose_reward, SV, fig_fn, choice = mf.plot_save(index,fn,data,parms,domain=domain,task=task,
        ylabel='prob_choose_lottery',xlabel='SV difference (SV_lottery - SV_fixed)',verbose=True)
    if not conf_drop:
        # if we keep the trials where there is no confidence measure we cannot store SV_hat for CASANDRE
        mf.store_SV(fn,crdm_df,SV,domain=domain,task=task,use_alpha=False)
    LL,LL0,AIC,BIC,R2,correct = mf.GOF_statistics(negLL,choice,p_choose_reward,nb_parms=3)
    p_range = max(p_choose_reward) - min(p_choose_reward)
    
    row = [subject,task.upper(),domain,response_rate,percent_lott,percent_risk,percent_ambig,
        conf_1,conf_2,conf_3,conf_4,negLL,gamma,beta,alpha,at_bound,LL,LL0,AIC,BIC,R2,
        correct,p_range,fig_fn]
    row_df = pd.DataFrame([row],columns=df_cols)
    return row_df
    

# can rewrite in terms of sort, fit, plot, like Corey Z does
def load_estimate_CRDM_save(split_dir='/tmp/',new_subjects=[],task='crdm',conf_drop=True,verbose=False):
    if verbose:
        print('We are working under /split_dir/ : {}'.format(split_dir))
    if conf_drop:
        print('\n **WARNING** The script will drop confidence responses that are either blank or None\n')
    crdm_files = mf.get_task_files(split_dir=split_dir,new_subjects=new_subjects,task=task)

    df_cols = ['subject','task','domain','response_rate','percent_lottery','percent_risk','percent_ambiguity',
        'conf_1','conf_2','conf_3','conf_4','negLL','gamma','beta','alpha','at_bound','LL','LL0',
        'AIC','BIC','R2','softmax_accuracy','softmax_range','fig_fn']
    df_out = pd.DataFrame(columns=df_cols)

    utility_dir = split_dir.replace('split','utility')
    make_dir(utility_dir)
    batch_name = mf.get_batch_name(split_dir=split_dir)
    df_dir = utility_dir
    df_fn = os.path.join(df_dir,'{}_CRDM_analysis.csv'.format(batch_name))

    # gamma, beta, alpha bounds
    # bounds determine by model_simulation
    gba_bounds = ((0,8),(-4.167,4.167),(0.125,4.341))
    counter = 0
    for index,fn in enumerate(crdm_files):
        # Load the CDD file and do some checks
        print('Working on CRDM csv file {} of {}:\n{}'.format(index+1,len(crdm_files),fn))
        subject = mf.get_subject(fn,task=task)
        df_orig = pd.read_csv(fn) #index_col=0 intentionally omitted

        # gain is always there
        domain = 'gain'
        crdm_df = mf.get_by_domain(df_orig,domain=domain,task=task,verbose=True)
        # fn_domain = fn.replace('.csv','_{}.csv'.format(domain)).replace('split','utility')
        # crdm_df.to_csv(fn_domain)
        row_df = estimate_CRDM_by_domain(crdm_df,fn,index,subject=subject,df_cols=df_cols,
                        gba_bounds = gba_bounds,domain=domain,task=task,conf_drop=conf_drop,
                        verbose=verbose)
        df_out = pd.concat([df_out,row_df],ignore_index=True)

        # domain_options = df_orig['crdm_domain'].dropna().unique()
        if 'loss' in df_orig['crdm_domain'].dropna().unique():
            domain = 'loss'                
            crdm_df = mf.get_by_domain(df_orig,domain=domain,task=task,verbose=True)
            # fn_domain = fn.replace('.csv','_{}.csv'.format(domain)).replace('split','utility')
            # crdm_df.to_csv(fn_domain)
            row_df = estimate_CRDM_by_domain(crdm_df,fn,index,subject=subject,df_cols=df_cols,
                            gba_bounds = gba_bounds,domain=domain,task=task,conf_drop=conf_drop,
                            verbose=verbose)
            df_out = pd.concat([df_out,row_df],ignore_index=True)

            domain = 'combined'
            row_df = estimate_CRDM_by_domain(df_orig,fn,index,subject=subject,df_cols=df_cols,
                            gba_bounds = gba_bounds,domain=domain,task=task,conf_drop=conf_drop,
                            verbose=verbose)
            df_out = pd.concat([df_out,row_df],ignore_index=True)            

        counter += 1

    # Save modeled parameters to modeled results
    mf.save_df_out(df_fn,df_out)

    return counter


def main():
    # if running this script on its own, start here
    # split_dir = '/Users/pizarror/mturk/idm_data/split'
    # one time hack

    # get paths to directories from the user
    split_dir = mf.get_split_dir()
    # SDAN_dir = '/Users/pizarror/mturk/idm_data/batch_output/SDAN'
    # split_dir = '/Users/pizarror/mturk/idm_data/batch_output/bonus2'
    conf_drop = False
    if 'SDM' in split_dir:
        conf_drop=True
    load_estimate_CRDM_save(split_dir=split_dir,conf_drop=conf_drop,verbose=True)


if __name__ == "__main__":
    main()










