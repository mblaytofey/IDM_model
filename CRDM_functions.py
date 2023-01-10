import pandas as pd
import numpy as np
import glob as glob
import os,sys
import scipy as sp
from scipy import optimize
import math
import matplotlib.pyplot as plt



def fit_ambiguity_risk_model(choice_sure_lott_amb, guesses = [1, 0.5, 0.5],
                             bkbounds = ((0,8),(1e-8,6.4),(1e-8,6.4)),disp=False):
    # We do start the optimizer off with the guesses.
    # The guesses are a starting point in parameter space for the optimizer. 

    # These are the bounds on betas. The first tuple corresponds to softmax slope, 
    # the second to beta on ambiguity, the third to value exponent, alpha.
    # most risky person to least risky person
    # most patient person >>> ???
    # least patient person >>> ???
    # beta0 = 0, Prob = 0.5
    # beta0 = 8 goes to infinity, prob=step function
    # bkbounds = ((0,8),(1e-8,6.4),(1e-8,6.4))

    # These are the inputs of the local_negLL function. They'll be passed through optimize_me()
    inputs = choice_sure_lott_amb.T.values.tolist()
    # print(inputs)

    # If seeking to improve performance, could change optimization method, could change maxiter(ations), 
    # or could fiddle with other things. 
    # You might be able to change the distance between steps in the optimzation.
    # results = optimize.minimize(optimize_me,guesses,inputs, bounds = bkbounds,method='L-BFGS-B', 
    #                             tol=1e-5, callback = None, options={'maxiter':10000, 'disp': True})
    results = optimize.minimize(optimize_me,guesses,inputs,bounds = bkbounds,
                                method='L-BFGS-B',options={'disp':disp})
    negLL = results.fun
    slope = results.x[0]
    beta = results.x[1]
    alpha = results.x[2]
    
    return negLL, slope, beta, alpha


def optimize_me(beta, inputs):
    choices_list,vF,vA,pF,pA,AL = inputs
    return local_negLL(beta,choices_list,vF,vA,pF,pA,AL)


def local_negLL(beta,choices_list,vF,vA,pF,pA,AL):

    ps_list,Util_fixed,Util_ambiguous = choice_prob_ambiguity_risk(vF,vA,pF,pA,AL,beta)
    ps = np.array(ps_list)
    choices = np.array(choices_list)

    # Trap log(0). This will prevent the code from trying to calculate the log of 0 in the next section.
    ps[ps==0] = 1e-6
    ps[ps==1] = 1-1e-6
    
    # Log-likelihood
    err = (choices==1)*np.log(ps) + ((choices==0))*np.log(1-ps)

    # Sum of -log-likelihood
    sumerr = -sum(err)

    return sumerr


def choice_prob_ambiguity_risk(vF,vA,pF,pA,AL,beta):
    ps = []
    utility_fixed_list = []
    utility_ambiguous_list = []
    ambig_fixed = 0
    for n in range(len(vF)):
        # subjective value (utility) fixed (sure bet)
        util_fix = ambiguity_utility(vF[n],pF[n],ambig_fixed,beta[2],beta[1])
        # subjective value (utility) ambiguous (lottery)
        util_amb = ambiguity_utility(vA[n],pA[n],AL[n],beta[2],beta[1])

        try: 
            p = 1 / (1 + math.exp(-beta[0]*(util_amb - util_fix)))
            # p = 1 / (1 + math.exp(beta_and_k_array[0]*(SS_SV-LL_SV)))     ## Math.exp does e^(). In other words, if the smaller-sooner SV is higher than the larger-later SV, e^x will be larger, making the denominator larger, making 1/denom closer to zero (low probability of choosing delay). If the LL SV is higher, the e^x will be lower, making 1/denom close to 1 (high probability of choosing delay). If they are the same, e^0=1, 1/(1+1) = 0.5, 50% chance of choosing delay.
        except OverflowError:                                             ## Sometimes the SS_SV is very much higher than the LL_SV. If beta gets too high, the exponent on e will get huge. Math.exp will throw an OverflowError if the numbers get too big. In that case, 1/(1+[something huge]) is essentially zero, so we just set it to 0.
            p = 0
        ps.append(p)
        utility_fixed_list.append(util_fix)
        utility_ambiguous_list.append(util_amb)
        
    return ps,utility_fixed_list,utility_ambiguous_list


def ambiguity_utility(v,p,AL,alpha,beta):
    if v>0:
        SV = (p - beta*AL/2) * (v**alpha)
    else:
        SV = (p - beta*AL/2) *(-1.0)*(abs(v)**alpha)
    return SV




def analysis(negLL,choices,ps,nb_parms=2):
    # Unrestricted log-likelihood
    LL = -negLL

    # Restricted log-likelihood
    LL0 = np.sum((choices==1)*math.log(0.5) + (1-(choices==1))*math.log(0.5))

    # Akaike Information Criterion
    AIC = -2*LL + 2*nb_parms  #CHANGE TO len(results.x) IF USING A DIFFERENT MODEL (parameters != 2)

    # Bayesian information criterion
    BIC = -2*LL + 2*math.log(len(ps))  #len(results.x)

    #R squared
    r2 = 1 - LL/LL0

    #Percent accuracy
    parray = np.array(ps) # gets an array of probabilities of choosing the LL choice
    correct =sum((parray>=0.5)==choices)/len(ps)                                          # LL is 1 in choices, so when the parray is > 0.5 and choices==1, the model has correctly predicted a choice.

    return LL,LL0,AIC,BIC,r2,correct

# Hessian unavailable in this optimization function, but would use results.hess_inv here
#Tester line if you want: print("LL",LL,"AIC",AIC,"BIC",BIC,"R2",r2,"correct",correct)

def main(args):
    print(args)

if __name__ == '__main__':
    main(sys.argv)

