import pandas as pd
import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from CDD_functions import fit_delay_discount_model,probability_choose_delay,store_SV
from CRDM_functions import GOF_statistics
from idm_split_data import make_dir
from scipy.interpolate import make_interp_spline, BSpline


def columns_there(df):
    cols_check = ['cdd_trial_resp.corr','cdd_immed_amt','cdd_immed_wait','cdd_delay_amt',
                  'cdd_delay_wait']
    for c in cols_check:
        if c not in list(df):
            print('Moving on to next subject, we could not find column : {}'.format(c))
            return 0
    return 1


def get_fig_fn(fn,use_alpha=False):
    fig_dir = os.path.dirname(fn).replace('idm_data/split/','figs/model/')
    make_dir(fig_dir)
    if use_alpha:
        fig_fn = os.path.join(fig_dir,os.path.basename(fn).replace('.csv','_model_fit_alpha.png'))
    else:
        fig_fn = os.path.join(fig_dir,os.path.basename(fn).replace('.csv','_model_fit.png'))

    return fig_fn


def plot_save(index,fn,data_choice_amt_wait,gamma,kappa,use_alpha=False,verbose=False):
    # extract values from dataframe to lists of values
    choice,value_soon,time_soon,value_delay,time_delay,alpha = data_choice_amt_wait.T.values.tolist()
    gamma_kappa = np.array([gamma,kappa])
    p_choose_delay,SV_soon,SV_delay = probability_choose_delay(value_soon,time_soon,value_delay,time_delay,gamma_kappa,alpha)
    SV_delta = [iSV_delay-iSV_soon for (iSV_delay,iSV_soon) in zip(SV_delay,SV_soon)]
    # for saving
    SV = SV_delta
    # sorted for plotting
    SV_delta, p_choose_delay, choice = zip(*sorted(zip(SV_delta, p_choose_delay, choice)))
    fig_fn = ''
    if gamma>0.001:
        SV_delta_new = np.linspace(min(SV_delta),max(SV_delta),300)
        SV_delta_x,p_choose_delay_y = zip(*set(zip(SV_delta, p_choose_delay)))
        SV_delta_x,p_choose_delay_y = zip(*sorted(zip(SV_delta_x,p_choose_delay_y)))
        spl = make_interp_spline(np.array(SV_delta_x),np.array(p_choose_delay_y),k=2)
        prob_smooth = spl(SV_delta_new)
        plt.figure(index)
        plt.plot(SV_delta_new,prob_smooth,'b-',linewidth=0.5)

        plt.plot(SV_delta,p_choose_delay,'b:',linewidth=1)
        plt.plot(SV_delta,choice,'r.')
        plt.plot([min(SV_delta),max(SV_delta)],[0.5,0.5],'k--',linewidth=0.5)
        plt.plot([0,0],[0.0,1.0],'k--',linewidth=0.5)
        plt.ylabel('prob_choose_delay')
        plt.xlabel('SV difference (SV_delay - SV_immediate)')
        fig_fn = get_fig_fn(fn,use_alpha=use_alpha)
        if verbose:
            print('Saving to : {}'.format(fig_fn))
        plt.savefig(fig_fn)
        plt.close(index)
    return p_choose_delay, SV, fig_fn, choice


def check_to_bound(gamma,kappa,gk_bounds= ((0,8),(1e-8,6.4))):
    at_bound = 0
    if gamma in gk_bounds[0]:
        at_bound = 1
    elif kappa  in gk_bounds[1]:
        at_bound = 1
    return at_bound


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


def load_estimate_CDD_save(split_dir='/tmp/',use_alpha=False,verbose=False):

    # Search for CDD files in the split_dir
    cdd_files = glob.glob(os.path.join(split_dir,'*/*/*_cdd.csv'))
    df_cols = ['subject','task','percent_impulse','negLL','gamma','kappa','alpha','at_bound','LL','LL0',
               'AIC','BIC','R2','correct','p_choose_delay_span','fig_fn']
    df_out = pd.DataFrame(columns=df_cols)

    df_dir = os.path.join(split_dir,'model_results')
    make_dir(df_dir)
    batch_name = os.path.basename(split_dir)
    df_fn = os.path.join(df_dir,'{}_CDD_analysis.csv'.format(batch_name))
    if use_alpha:
        df_fn = df_fn.replace('.csv','_alpha.csv')

    gk_bounds = ((0,8),(1e-3,8))
    for index,fn in enumerate(cdd_files):
        # Load the CDD file and do some checks
        print('Working on the following CDD csv file :\n{}'.format(fn))
        subject = os.path.basename(fn).replace('_cdd.csv','')
        cdd_df = pd.read_csv(fn) #index_col=0 intentionally avoided
        if not columns_there(cdd_df):
            continue
        
        # default value if not using alpha for modeling
        alpha_hat=1
        if use_alpha:
            alpha_hat = get_alpha_hat(model_dir=df_dir,batch_name=batch_name,subject=subject)
        cols = ['cdd_trial_resp.corr','cdd_immed_amt','cdd_immed_wait','cdd_delay_amt','cdd_delay_wait']
        data_choice_amt_wait, percent_impulse = get_data(cdd_df,cols,alpha_hat=alpha_hat)
        # Estimate gamma and kappa with or without alpha
        negLL,gamma,kappa = fit_delay_discount_model(data_choice_amt_wait,
                                                          gk_guess = [0.15, 0.5],
                                                          gk_bounds = gk_bounds, disp=False)
        at_bound = check_to_bound(gamma,kappa,gk_bounds=gk_bounds)
        if verbose:
            print('From CRDM we estimated the following alpha value : {}'.format(alpha_hat))
            print('Percent Impulse Choice: {}'.format(percent_impulse))
            print("Negative log-likelihood: {}, gamma: {}, kappa: {}".
                  format(negLL, gamma, kappa))

        p_choose_delay, SV, fig_fn, choice = plot_save(index,fn,data_choice_amt_wait,gamma,kappa,use_alpha=use_alpha)
        store_SV(fn,cdd_df,SV_delta=SV,task='cdd',use_alpha=use_alpha)
        LL,LL0,AIC,BIC,R2,correct = GOF_statistics(negLL,choice,p_choose_delay,nb_parms=2)
        p_choose_delay_range = max(p_choose_delay) - min(p_choose_delay)
        
        row = [subject,'cdd',percent_impulse,negLL,gamma,kappa,alpha_hat,at_bound,LL,LL0,AIC,BIC,R2,correct,p_choose_delay_range,fig_fn]
        row_df = pd.DataFrame([row],columns=df_cols)
        df_out = pd.concat([df_out,row_df],ignore_index=True)

    # Save modeled parameters to modeled results
    print('Saving analysis to : {}'.format(df_fn))
    df_out.to_csv(df_fn)



def main():
    # if running this script on its own, start here
    split_dir = '/Users/pizarror/mturk/idm_data/split'
    load_estimate_CDD_save(split_dir)



if __name__ == "__main__":
    main()










