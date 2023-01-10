import pandas as pd
import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from CRDM_functions import fit_ambiguity_risk_model,choice_prob_ambiguity_risk,analysis
from idm_split_data import make_dir
from scipy.interpolate import make_interp_spline, BSpline

def columns_there(df):
    cols_check = ['crdm_trial_resp.corr','crdm_sure_amt','crdm_lott_top','crdm_lott_bot',
                  'crdm_sure_p','crdm_lott_p','crdm_amb_lev']
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

def plot_save(index,fn,choice_sure_lott_amb,slope,beta,alpha):

    choices_list,vF,vA,pF,pA,AL = choice_sure_lott_amb.T.values.tolist()
    beta_array = np.array([slope,beta,alpha])
    ps,utility_fixed_list,utility_ambiguous_list = choice_prob_ambiguity_risk(
            vF,vA,pF,pA,AL,beta_array)
    util_delta = [amb-fix for (amb,fix) in zip(utility_ambiguous_list,utility_fixed_list)]
    # print(list(zip(SV_delta,ps)))
    # sys.exit()
    util_delta, ps, choices_list = zip(*sorted(zip(util_delta, ps, choices_list)))
    fig_fn = ''
    if slope>0.001:
        util_delta_new = np.linspace(min(util_delta),max(util_delta),300)
        util_delta_x,ps_y = zip(*set(zip(util_delta, ps)))
        util_delta_x,ps_y = zip(*sorted(zip(util_delta_x,ps_y)))
        spl = make_interp_spline(np.array(util_delta_x),np.array(ps_y),k=2)
        prob_smooth = spl(util_delta_new)
        plt.figure(index)
        plt.plot(util_delta_new,prob_smooth,'b-',linewidth=0.5)
        plt.plot(util_delta,ps,'b:',linewidth=1)
        plt.plot(util_delta,choices_list,'r.')
        plt.plot([min(util_delta),max(util_delta)],[0.5,0.5],'k--',linewidth=0.5)
        plt.plot([0,0],[0.0,1.0],'k--',linewidth=0.5)
        plt.ylabel('probability softmax')
        plt.xlabel('SV difference (SV_lottery - SV_fixed)')
        fig_fn = get_fig_fn(fn)
        print('Saving to : {}'.format(fig_fn))
        plt.savefig(fig_fn)
        plt.close(index)
    return ps, fig_fn, choices_list


def check_to_bound(slope,beta,alpha,bkbounds= ((0,8),(1e-8,6.4),(1e-8,6.4))):
    at_bound = 0
    if slope in bkbounds[0]:
        at_bound = 1
    elif beta in bkbounds[1]:
        at_bound = 1
    elif alpha  in bkbounds[2]:
        at_bound = 1
    return at_bound

def main():
    split_dir = '/Users/pizarror/mturk/idm_data/split'
    crdm_files = glob.glob(os.path.join(split_dir,'*/*/*_crdm.csv'))
    df_cols = ['subject','task','percent_risk','negLL','slope','beta','alpha','at_bound','LL','LL0',
               'AIC','BIC','R2','correct','prob_span','fig_fn']
    df_out = pd.DataFrame(columns=df_cols)
    bkbounds = ((0,8),(1e-8,6.4),(1e-8,6.4))
    for index,fn in enumerate(crdm_files):
        print(fn)
        subj = os.path.basename(fn).replace('_crdm.csv','')
        crdm_df = pd.read_csv(fn)#index_col=0 intentionally avoided
        if not columns_there(crdm_df):
            continue
        crdm_df['crdm_lott_amt'] = crdm_df['crdm_lott_top']+crdm_df['crdm_lott_bot']
        cols = ['crdm_trial_resp.corr','crdm_sure_amt','crdm_lott_amt',
                'crdm_sure_p','crdm_lott_p','crdm_amb_lev']
        # choices_list,vF,vA,pF,pA,AL
        choice_sure_lott_amb = crdm_df[cols]
        choice_sure_lott_amb = choice_sure_lott_amb.dropna()
        # resp.corr = 0 is lottery, resp.corr = 1 is safe $5
        choice_sure_lott_amb['crdm_trial_resp.corr'] = 1.0 - choice_sure_lott_amb['crdm_trial_resp.corr']
        # print(choice_sure_lott_amb)
        perc_risk = 1.0 - 1.0*choice_sure_lott_amb['crdm_trial_resp.corr'].sum()/choice_sure_lott_amb['crdm_trial_resp.corr'].shape[0]
        print('Percent Risky Choice: {}'.format(perc_risk))

        negLL,slope,beta,alpha = fit_ambiguity_risk_model(choice_sure_lott_amb,
                                                          guesses = [0.15, 0.5, 0.5],
                                                          bkbounds = bkbounds,disp=False)
        at_bound = check_to_bound(slope,beta,alpha,bkbounds=bkbounds)
        print("Negative log-likelihood: {}, slope: {}, beta: {}, alpha: {}".
              format(negLL, slope,beta, alpha))

        ps, fig_fn, choices_list = plot_save(index,fn,choice_sure_lott_amb,slope,beta,alpha)
        LL,LL0,AIC,BIC,R2,correct = analysis(negLL,choices_list,ps,nb_parms=3)
        ps_range = max(ps) - min(ps)
        if ps_range>0.6:
            print(correct)
            print(list(zip(ps,choices_list)))
        
        row = [subj,'CRDM',perc_risk,negLL,slope,beta,alpha,at_bound,LL,LL0,AIC,BIC,R2,correct,ps_range,fig_fn]
        row_df = pd.DataFrame([row],columns=df_cols)
        df_out = pd.concat([df_out,row_df],ignore_index=True)
    print(df_out)
    df_fn = '/Users/pizarror/mturk/model_results/CRDM_analysis.csv'
    print('Saving analysis to : {}'.format(df_fn))
    df_out.to_csv(df_fn)

if __name__ == "__main__":
    main()










