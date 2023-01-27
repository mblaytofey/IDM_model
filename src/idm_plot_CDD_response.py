import pandas as pd
import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from CDD_functions import fit_delay_discount_model,probability_choose_delay
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
    # print('All columns present')
    return 1


def get_fig_fn(fn):
    fig_dir = os.path.dirname(fn).replace('idm_data/split/','figs/model/')
    make_dir(fig_dir)
    fig_fn = os.path.join(fig_dir,os.path.basename(fn).replace('.csv','_model_fit.png'))
    return fig_fn


def plot_save(index,fn,data_choice_amt_wait,gamma,kappa):
    # extract values from dataframe to lists of values
    choice,value_soon,time_soon,value_delay,time_delay,risk = data_choice_amt_wait.T.values.tolist()
    gamma_kappa = np.array([gamma,kappa])
    p_choose_delay,SV_soon,SV_delay = probability_choose_delay(value_soon,time_soon,value_delay,time_delay,gamma_kappa,risk)
    SV_delta = [iSV_delay-iSV_soon for (iSV_delay,iSV_soon) in zip(SV_delay,SV_soon)]
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
        fig_fn = get_fig_fn(fn)
        print('Saving to : {}'.format(fig_fn))
        plt.savefig(fig_fn)
        plt.close(index)
    return p_choose_delay, fig_fn, choice


def check_to_bound(gamma,kappa,gk_bounds= ((0,8),(1e-8,6.4))):
    at_bound = 0
    if gamma in gk_bounds[0]:
        at_bound = 1
    elif kappa  in gk_bounds[1]:
        at_bound = 1
    return at_bound


def get_data(df,cols):
    # select from columns
    data = df[cols]
    # drop rows with NA int them
    data = data.dropna()
    # add risk column, will change later
    data['risk']=1.0
    # resp.corr = 0 is lottery, resp.corr = 1 is safe $5
    data['cdd_trial_resp.corr'] = 1.0 - data['cdd_trial_resp.corr']
    percent_impulse = 1.0 - 1.0*data['cdd_trial_resp.corr'].sum()/data['cdd_trial_resp.corr'].shape[0]
    return data,percent_impulse


def load_estimate_CDD_save(split_dir='/tmp/',alpha=False):

    cdd_files = glob.glob(os.path.join(split_dir,'*/*/*_cdd.csv'))
    df_cols = ['subject','task','percent_impulse','negLL','gamma','kappa','at_bound','LL','LL0',
               'AIC','BIC','R2','correct','p_choose_delay_span','fig_fn']
    df_out = pd.DataFrame(columns=df_cols)
    gk_bounds = ((0,8),(1e-8,6.4))
    for index,fn in enumerate(cdd_files):
        print(fn)
        subj = os.path.basename(fn).replace('_cdd.csv','')
        cdd_df = pd.read_csv(fn) #index_col=0 intentionally avoided
        if not columns_there(cdd_df):
            continue

        cols = ['cdd_trial_resp.corr','cdd_immed_amt','cdd_immed_wait','cdd_delay_amt','cdd_delay_wait']
        data_choice_amt_wait, percent_impulse = get_data(cdd_df,cols)
        print('Percent Impulse Choice: {}'.format(percent_impulse))

        negLL,gamma,kappa,risk = fit_delay_discount_model(data_choice_amt_wait,
                                                          gk_guess = [0.15, 0.5],
                                                          gk_bounds = gk_bounds,disp=False)
        at_bound = check_to_bound(gamma,kappa,gk_bounds=gk_bounds)
        print("Negative log-likelihood: {}, gamma: {}, kappa: {}".
              format(negLL, gamma, kappa))

        p_choose_delay, fig_fn, choice = plot_save(index,fn,data_choice_amt_wait,gamma,kappa)
        LL,LL0,AIC,BIC,R2,correct = GOF_statistics(negLL,choice,p_choose_delay,nb_parms=2)
        p_choose_delay_range = max(p_choose_delay) - min(p_choose_delay)
        
        row = [subj,'cdd',percent_impulse,negLL,gamma,kappa,at_bound,LL,LL0,AIC,BIC,R2,correct,p_choose_delay_range,fig_fn]
        row_df = pd.DataFrame([row],columns=df_cols)
        df_out = pd.concat([df_out,row_df],ignore_index=True)
    print(df_out)
    df_dir = os.path.join(split_dir,'model_results')
    make_dir(df_dir)
    batch_name = os.path.basename(split_dir)
    df_fn = os.path.join(df_dir,'{}_CDD_analysis.csv'.format(batch_name))
    # df_fn = '/Users/pizarror/mturk/model_results/CDD_analysis.csv'
    print('Saving analysis to : {}'.format(df_fn))
    df_out.to_csv(df_fn)



def main():
    # if running this script on its own, start here
    split_dir = '/Users/pizarror/mturk/idm_data/split'
    load_estimate_CDD_save(split_dir)



if __name__ == "__main__":
    main()










