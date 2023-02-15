import pandas as pd
import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from CRDM_functions import fit_ambiguity_risk_model,probability_choose_ambiguity,GOF_statistics,get_data, drop_non_responses, get_task_files
from CDD_functions import store_SV
from idm_split_data import make_dir
from scipy.interpolate import make_interp_spline, BSpline


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



def get_fig_fn(fn):
    # fig_dir = os.path.dirname(fn) #.replace('idm_data/split/','figs/model/')
    split_dir = os.path.dirname(os.path.dirname(os.path.dirname(fn)))
    # make_dir(fig_dir)
    fig_fn = fn.replace(split_dir,'').replace('.csv','_model_fit.png')[1:]
    # fig_fn = os.path.join(fig_dir,os.path.basename(fn).replace('.csv','_model_fit.png'))
    return split_dir,fig_fn


def plot_save(index,fn,data_choice_sure_lott_amb,gamma,beta,alpha,verbose=False):
    # extract values from dataframe to lists of values
    choice,value_fix,value_ambig,p_fix,p_ambig,ambiguity = data_choice_sure_lott_amb.T.values.tolist()
    gamma_beta_alpha = np.array([gamma,beta,alpha])
    p_choose_ambig,SV_fix,SV_ambig = probability_choose_ambiguity(value_fix,value_ambig,p_fix,p_ambig,ambiguity,gamma_beta_alpha)
    SV_delta = [amb-fix for (amb,fix) in zip(SV_ambig,SV_fix)]
    # for saving
    SV = SV_delta
    # sorted for plotting
    SV_delta, p_choose_ambig, choice = zip(*sorted(zip(SV_delta, p_choose_ambig, choice)))

    split_dir,fig_fn = get_fig_fn(fn)
    plt.figure(index)
    SV_delta_new = np.linspace(min(SV_delta),max(SV_delta),300)
    SV_delta_x,p_choose_ambig_y = zip(*set(zip(SV_delta, p_choose_ambig)))
    SV_delta_x,p_choose_ambig_y = zip(*sorted(zip(SV_delta_x,p_choose_ambig_y)))
    spl = make_interp_spline(np.array(SV_delta_x),np.array(p_choose_ambig_y),k=2)
    prob_smooth = spl(SV_delta_new)
    plt.plot(SV_delta_new,prob_smooth,'b-',linewidth=0.5)
    plt.plot(SV_delta,p_choose_ambig,'b:',linewidth=1)
    plt.plot(SV_delta,choice,'r*-',linewidth=0.5)
    plt.plot([min(SV_delta),max(SV_delta)],[0.5,0.5],'k--',linewidth=0.5)
    plt.plot([0,0],[0.0,1.0],'k--',linewidth=0.5)
    plt.ylabel('prob_choose_ambig',fontsize=12)
    plt.xlabel('SV difference (SV_lottery - SV_fixed)',fontsize=12)
    if verbose:
        plt.title(get_subject(fn),fontsize=15)
        print('Saving to : /split_dir/ {}'.format(fig_fn))
    plt.savefig(os.path.join(split_dir,fig_fn))
    plt.close(index)
    return p_choose_ambig, SV,fig_fn, choice


def check_to_bound(gamma,beta,alpha,gba_bounds= ((0,8),(1e-8,6.4),(1e-8,6.4))):
    at_bound = 0
    if gamma in gba_bounds[0]:
        at_bound = 1
    elif beta in gba_bounds[1]:
        at_bound = 1
    elif alpha  in gba_bounds[2]:
        at_bound = 1
    return at_bound


def get_subject(fn):
    return os.path.basename(fn).replace('_crdm.csv','')

def load_estimate_CRDM_save(split_dir='/tmp/', verbose=False):
    if verbose:
        print('We are working under /split_dir/ : {}'.format(split_dir))
    crdm_files = get_task_files(split_dir=split_dir,task='crdm')

    df_cols = ['subject','task','percent_risk','negLL','gamma','beta','alpha','at_bound','LL','LL0',
               'AIC','BIC','R2','correct','p_choose_ambig_span','fig_fn']
    df_out = pd.DataFrame(columns=df_cols)

    df_dir = split_dir
    batch_name = os.path.basename(split_dir)
    df_fn = os.path.join(df_dir,'{}_CRDM_analysis.csv'.format(batch_name))

    # gamma, beta, alpha bounds
    gba_bounds = ((0,8),(1e-8,6.4),(0.125,4.341))
    counter = 0
    for index,fn in enumerate(crdm_files):
        # Load the CDD file and do some checks
        print('Working on the following CRDM csv file :\n{}'.format(fn))
        subject = get_subject(fn)
        crdm_df = pd.read_csv(fn) #index_col=0 intentionally omitted
        crdm_df = drop_non_responses(crdm_df)
        # crdm_df = crdm_df[crdm_df['crdm_conf_resp.rt'].notna()]

        if not columns_there(crdm_df):
            # hack for columns not being named properly, check again
            print('Checking if renaming columns work')
            crdm_df = rename_columns(crdm_df)
            if not columns_there(crdm_df):
                print('Tried renaming and did not work, check .csv file and try again')
                continue
            elif verbose:
                print('Renaming worked, we will continue as such')
                print(crdm_df)

        cols = ['crdm_trial_resp.corr','crdm_sure_amt','crdm_lott_amt','crdm_sure_p','crdm_lott_p','crdm_amb_lev']
        data_choice_sure_lott_amb,percent_risk = get_data(crdm_df,cols)
        # Estimate gamma, beta, and alpha
        negLL,gamma,beta,alpha = fit_ambiguity_risk_model(data_choice_sure_lott_amb,
                                                          gba_guess = [0.15, 0.5, 0.6],
                                                          gba_bounds = gba_bounds,disp=False)
        at_bound = check_to_bound(gamma,beta,alpha,gba_bounds=gba_bounds)
        if verbose:
            print('Percent Risky Choice: {}'.format(percent_risk))
            print("Negative log-likelihood: {}, gamma: {}, beta: {}, alpha: {}".
                  format(negLL, gamma, beta, alpha))

        p_choose_ambig, SV, fig_fn, choice = plot_save(index,fn,data_choice_sure_lott_amb,gamma,beta,alpha,verbose=verbose)
        store_SV(fn,crdm_df,SV,task='crdm',use_alpha=False)
        LL,LL0,AIC,BIC,R2,correct = GOF_statistics(negLL,choice,p_choose_ambig,nb_parms=3)
        p_choose_ambig_range = max(p_choose_ambig) - min(p_choose_ambig)
        
        row = [subject,'CRDM',percent_risk,negLL,gamma,beta,alpha,at_bound,LL,LL0,AIC,BIC,R2,correct,p_choose_ambig_range,fig_fn]
        row_df = pd.DataFrame([row],columns=df_cols)
        df_out = pd.concat([df_out,row_df],ignore_index=True)

        counter += 1

    # Save modeled parameters to modeled results
    print('Saving analysis to : {}'.format(df_fn))
    df_out.to_csv(df_fn)

    return counter


def main():
    # if running this script on its own, start here
    # split_dir = '/Users/pizarror/mturk/idm_data/split'
    # one time hack
    # SDAN_dir = '/Users/pizarror/mturk/idm_data/batch_output/SDAN'
    split_dir = '/Users/pizarror/mturk/idm_data/batch_output/bonus2'
    load_estimate_CRDM_save(split_dir,verbose=True)


if __name__ == "__main__":
    main()










