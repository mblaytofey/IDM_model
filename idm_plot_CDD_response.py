import pandas as pd
import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from CDD_functions import fit_delay_discount_model,choice_prob
from CRDM_functions import analysis
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

def plot_save(index,fn,amt_wait_choice,beta,kappa):

    choices_list,SS_V,SS_D,LL_V,LL_D,risk = amt_wait_choice.T.values.tolist()
    beta_and_k_array = np.array([beta,kappa])
    ps,SS_SV,LL_SV = choice_prob(SS_V,SS_D,LL_V,LL_D,beta_and_k_array,risk)
    SV_delta = [ll-ss for (ll,ss) in zip(LL_SV,SS_SV)]
    SV_delta, ps, choices_list = zip(*sorted(zip(SV_delta, ps, choices_list)))
    fig_fn = ''
    if beta>0.001:
        SV_delta_new = np.linspace(min(SV_delta),max(SV_delta),300)
        SV_delta_x,ps_y = zip(*set(zip(SV_delta, ps)))
        SV_delta_x,ps_y = zip(*sorted(zip(SV_delta_x,ps_y)))
        spl = make_interp_spline(np.array(SV_delta_x),np.array(ps_y),k=2)
        prob_smooth = spl(SV_delta_new)
        plt.figure(index)
        plt.plot(SV_delta_new,prob_smooth,'b-',linewidth=0.5)
        plt.plot(SV_delta,ps,'b:',linewidth=1)
        plt.plot(SV_delta,choices_list,'r.')
        plt.plot([min(SV_delta),max(SV_delta)],[0.5,0.5],'k--',linewidth=0.5)
        plt.plot([0,0],[0.0,1.0],'k--',linewidth=0.5)
        plt.ylabel('probability softmax')
        plt.xlabel('SV difference (SV_delay - SV_immediate)')
        fig_fn = get_fig_fn(fn)
        print('Saving to : {}'.format(fig_fn))
        plt.savefig(fig_fn)
        plt.close(index)
    return ps, fig_fn, choices_list


def check_to_bound(beta,kappa,bkbounds= ((0,8),(1e-8,6.4))):
    at_bound = 0
    if beta in bkbounds[0]:
        at_bound = 1
    elif kappa  in bkbounds[1]:
        at_bound = 1
    return at_bound

def main():
    split_dir = '/Users/pizarror/mturk/idm_data/split'
    cdd_files = glob.glob(os.path.join(split_dir,'*/*/*_cdd.csv'))
    df_cols = ['subject','task','percent_impulse','negLL','beta','kappa','at_bound','LL','LL0',
               'AIC','BIC','R2','correct','prob_span','fig_fn']
    df_out = pd.DataFrame(columns=df_cols)
    bkbounds = ((0,8),(1e-8,6.4))
    for index,fn in enumerate(cdd_files):
        print(fn)
        subj = os.path.basename(fn).replace('_cdd.csv','')
        cdd_df = pd.read_csv(fn)#index_col=0 intentionally avoided
        if not columns_there(cdd_df):
            continue
        cols = ['cdd_trial_resp.corr','cdd_immed_amt','cdd_immed_wait','cdd_delay_amt',
                'cdd_delay_wait']
        # choices_list,vF,vA,pF,pA,AL
        amt_wait_choice = cdd_df[cols]
        amt_wait_choice = amt_wait_choice.dropna()
        amt_wait_choice['risk']=1.0
        # resp.corr = 0 is lottery, resp.corr = 1 is safe $5
        amt_wait_choice['cdd_trial_resp.corr'] = 1.0 - amt_wait_choice['cdd_trial_resp.corr']
        # print(amt_wait_choice)
        perc_impulse = 1.0 - 1.0*amt_wait_choice['cdd_trial_resp.corr'].sum()/amt_wait_choice['cdd_trial_resp.corr'].shape[0]
        print('Percent Impulse Choice: {}'.format(perc_impulse))

        negLL,beta,kappa,risk = fit_delay_discount_model(amt_wait_choice,
                                                          guesses = [0.15, 0.5],
                                                          bkbounds = bkbounds,disp=False)
        at_bound = check_to_bound(beta,kappa,bkbounds=bkbounds)
        print("Negative log-likelihood: {}, beta: {}, kappa: {}".
              format(negLL, beta, kappa))

        ps, fig_fn, choices_list = plot_save(index,fn,amt_wait_choice,beta,kappa)
        LL,LL0,AIC,BIC,R2,correct = analysis(negLL,choices_list,ps,nb_parms=2)
        ps_range = max(ps) - min(ps)
        if ps_range>0.6:
            print(correct)
            print(list(zip(ps,choices_list)))
        
        row = [subj,'cdd',perc_impulse,negLL,beta,kappa,at_bound,LL,LL0,AIC,BIC,R2,correct,ps_range,fig_fn]
        row_df = pd.DataFrame([row],columns=df_cols)
        df_out = pd.concat([df_out,row_df],ignore_index=True)
    print(df_out)
    df_fn = '/Users/pizarror/mturk/model_results/cdd_analysis.csv'
    print('Saving analysis to : {}'.format(df_fn))
    df_out.to_csv(df_fn)

if __name__ == "__main__":
    main()










