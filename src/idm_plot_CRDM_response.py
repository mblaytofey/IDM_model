import pandas as pd
import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from CRDM_functions import fit_ambiguity_risk_model,probability_choose_ambiguity,GOF_statistics
from idm_split_data import make_dir
from scipy.interpolate import make_interp_spline, BSpline


def columns_there(df):
    cols_check = ['crdm_trial_resp.corr','crdm_sure_amt','crdm_lott_top','crdm_lott_bot',
                  'crdm_sure_p','crdm_lott_p','crdm_amb_lev']
    for c in cols_check:
        if c not in list(df):
            print('Moving on to next subject, we could not find column : {}'.format(c))
            return 0
    return 1


def get_fig_fn(fn):
    fig_dir = os.path.dirname(fn).replace('idm_data/split/','figs/model/')
    make_dir(fig_dir)
    fig_fn = os.path.join(fig_dir,os.path.basename(fn).replace('.csv','_model_fit.png'))
    return fig_fn


def plot_save(index,fn,data_choice_sure_lott_amb,gamma,beta,alpha):
    # extract values from dataframe to lists of values
    choice,value_fix,value_ambig,p_fix,p_ambig,ambiguity = data_choice_sure_lott_amb.T.values.tolist()
    gamma_beta_alpha = np.array([gamma,beta,alpha])
    p_choose_ambig,SV_fix,SV_ambig = probability_choose_ambiguity(value_fix,value_ambig,p_fix,p_ambig,ambiguity,gamma_beta_alpha)
    SV_delta = [amb-fix for (amb,fix) in zip(SV_ambig,SV_fix)]
    SV_delta, p_choose_ambig, choice = zip(*sorted(zip(SV_delta, p_choose_ambig, choice)))
    fig_fn = ''
    if gamma>0.001:
        SV_delta_new = np.linspace(min(SV_delta),max(SV_delta),300)
        SV_delta_x,p_choose_ambig_y = zip(*set(zip(SV_delta, p_choose_ambig)))
        SV_delta_x,p_choose_ambig_y = zip(*sorted(zip(SV_delta_x,p_choose_ambig_y)))
        spl = make_interp_spline(np.array(SV_delta_x),np.array(p_choose_ambig_y),k=2)
        prob_smooth = spl(SV_delta_new)
        plt.figure(index)
        plt.plot(SV_delta_new,prob_smooth,'b-',linewidth=0.5)

        plt.plot(SV_delta,p_choose_ambig,'b:',linewidth=1)
        plt.plot(SV_delta,choice,'r.')
        plt.plot([min(SV_delta),max(SV_delta)],[0.5,0.5],'k--',linewidth=0.5)
        plt.plot([0,0],[0.0,1.0],'k--',linewidth=0.5)
        plt.ylabel('prob_choose_ambig')
        plt.xlabel('SV difference (SV_lottery - SV_fixed)')
        fig_fn = get_fig_fn(fn)
        print('Saving to : {}'.format(fig_fn))
        plt.savefig(fig_fn)
        plt.close(index)
    return p_choose_ambig, fig_fn, choice


def check_to_bound(gamma,beta,alpha,gba_bounds= ((0,8),(1e-8,6.4),(1e-8,6.4))):
    at_bound = 0
    if gamma in gba_bounds[0]:
        at_bound = 1
    elif beta in gba_bounds[1]:
        at_bound = 1
    elif alpha  in gba_bounds[2]:
        at_bound = 1
    return at_bound


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


def load_estimate_CRDM_save(split_dir='/tmp/'):

    crdm_files = glob.glob(os.path.join(split_dir,'*/*/*_crdm.csv'))
    df_cols = ['subject','task','percent_risk','negLL','gamma','beta','alpha','at_bound','LL','LL0',
               'AIC','BIC','R2','correct','p_choose_ambig_span','fig_fn']
    df_out = pd.DataFrame(columns=df_cols)
    # gamma, beta, alpha bounds
    gba_bounds = ((0,8),(1e-8,6.4),(1e-8,6.4))
    counter = 0
    for index,fn in enumerate(crdm_files):
        print(fn)
        subj = os.path.basename(fn).replace('_crdm.csv','')
        crdm_df = pd.read_csv(fn) #index_col=0 intentionally avoided
        if not columns_there(crdm_df):
            continue

        cols = ['crdm_trial_resp.corr','crdm_sure_amt','crdm_lott_amt','crdm_sure_p','crdm_lott_p','crdm_amb_lev']
        data_choice_sure_lott_amb,percent_risk = get_data(crdm_df,cols)
        print('Percent Risky Choice: {}'.format(percent_risk))

        negLL,gamma,beta,alpha = fit_ambiguity_risk_model(data_choice_sure_lott_amb,
                                                          gba_guess = [0.15, 0.5, 0.5],
                                                          gba_bounds = gba_bounds,disp=False)
        at_bound = check_to_bound(gamma,beta,alpha,gba_bounds=gba_bounds)
        print("Negative log-likelihood: {}, gamma: {}, beta: {}, alpha: {}".
              format(negLL, gamma, beta, alpha))

        p_choose_ambig, fig_fn, choice = plot_save(index,fn,data_choice_sure_lott_amb,gamma,beta,alpha)
        LL,LL0,AIC,BIC,R2,correct = GOF_statistics(negLL,choice,p_choose_ambig,nb_parms=3)
        p_choose_ambig_range = max(p_choose_ambig) - min(p_choose_ambig)
        
        row = [subj,'CRDM',percent_risk,negLL,gamma,beta,alpha,at_bound,LL,LL0,AIC,BIC,R2,correct,p_choose_ambig_range,fig_fn]
        row_df = pd.DataFrame([row],columns=df_cols)
        df_out = pd.concat([df_out,row_df],ignore_index=True)

        counter += 1

    total_modeled=True
    if counter<index:
        # We did not anaklyze all CRDM files, have to check
        print('For some reason we only modeled CRDM for {} of {} files, please check the log files'.format(counter,index))
        total_modeled=False

    df_dir = os.path.join(split_dir,'model_results')
    make_dir(df_dir)
    batch_name = os.path.basename(split_dir)
    df_fn = os.path.join(df_dir,'{}_CRDM_analysis.csv'.format(batch_name))
    print('Saving analysis to : {}'.format(df_fn))
    df_out.to_csv(df_fn)

    return total_modeled,counter


def main():
    # if running this script on its own, start here
    split_dir = '/Users/pizarror/mturk/idm_data/split'
    load_estimate_CRDM_save(split_dir)


if __name__ == "__main__":
    main()










