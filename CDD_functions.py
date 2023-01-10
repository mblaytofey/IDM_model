import pandas as pd
import numpy as np
import glob as glob
import os,sys
import scipy as sp
from scipy import optimize
import math
import matplotlib.pyplot as plt



def fit_delay_discount_model(amt_wait_choice, guesses = [0.15, 0.5],bkbounds = ((0,8),(1e-8,6.4)),disp=False):
    # We do start the optimizer off with the guesses below, but those aren't updated like Bayesian priors. 
    # They are simply a starting point in parameter space for the optimizer. Changes here could be an avenue 
    # to explore when seeking to improve performance.
    # [beta, kappa]
    # guesses = [0.15, 0.5]

    # These are the bounds on k and beta. The first tuple corresponds to beta, the second to kappa.
    # (beta, kappa) 
    # most impulsive person to least impulsive person
    # most patient person >>> lowest kappa
    # least patient person >>> highest kappa
    # beta = 0, Prob = 0.5
    # beta = 8 goes to infinity, prob=step function
    # bkbounds = ((0,8),(1e-8,6.4))
    risk = 1
    # These are the inputs of the local_negLL function. They'll be passed through optimize_me()

    inputs = amt_wait_choice.T.values.tolist()
    # print(inputs)

    # If seeking to improve performance, could change optimization method, could change maxiter(ations), 
    # or could fiddle with other things. 
    # You might be able to change the distance between steps in the optimzation.
    # results = optimize.minimize(optimize_me,guesses,inputs, bounds = bkbounds,method='L-BFGS-B', 
    #                             tol=1e-5, callback = None, options={'maxiter':10000, 'disp': True})
    results = optimize.minimize(optimize_me,guesses,inputs,bounds = bkbounds,
                                method='L-BFGS-B',options={'disp':disp})
    negLL = results.fun
    beta = results.x[0]
    k = results.x[1]
    
    return negLL, beta, k, risk


def optimize_me(beta_and_k_array_to_optimize, inputs):
    choices_list,SS_V,SS_D,LL_V,LL_D,risk = inputs
    return local_negLL(beta_and_k_array_to_optimize,choices_list,SS_V,SS_D,LL_V,LL_D,risk)


def local_negLL(beta_and_k_array,choices_list,SS_V,SS_D,LL_V,LL_D,risk):

    ps_list,SS_SV,LL_SV = choice_prob(SS_V,SS_D,LL_V,LL_D,beta_and_k_array,risk)
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


def choice_prob(SS_V,SS_D,LL_V,LL_D,beta_and_k_array,risk):
    ps = []
    SS_SV_list = []
    LL_SV_list = []
    for n in range(len(SS_V)):
        # smaller sooner SV_now
        SS_SV = discount(SS_V[n],SS_D[n],beta_and_k_array[1],risk[n])
        # larger later SV_delay
        LL_SV = discount(LL_V[n],LL_D[n],beta_and_k_array[1],risk[n])

        try: 
            p = 1 / (1 + math.exp(-beta_and_k_array[0]*(LL_SV-SS_SV)))
            # p = 1 / (1 + math.exp(beta_and_k_array[0]*(SS_SV-LL_SV)))     ## Math.exp does e^(). In other words, if the smaller-sooner SV is higher than the larger-later SV, e^x will be larger, making the denominator larger, making 1/denom closer to zero (low probability of choosing delay). If the LL SV is higher, the e^x will be lower, making 1/denom close to 1 (high probability of choosing delay). If they are the same, e^0=1, 1/(1+1) = 0.5, 50% chance of choosing delay.
        except OverflowError:                                             ## Sometimes the SS_SV is very much higher than the LL_SV. If beta gets too high, the exponent on e will get huge. Math.exp will throw an OverflowError if the numbers get too big. In that case, 1/(1+[something huge]) is essentially zero, so we just set it to 0.
            p = 0
        ps.append(p)
        SS_SV_list.append(SS_SV)
        LL_SV_list.append(LL_SV)
        
    return ps,SS_SV_list,LL_SV_list


def discount(v,d,kappa,risk):
    SV = (v**risk)/(1+kappa*d)
    return SV


# Hessian unavailable in this optimization function, but would use results.hess_inv here
#Tester line if you want: print("LL",LL,"AIC",AIC,"BIC",BIC,"R2",r2,"correct",correct)

def main(args):
    print(args)

if __name__ == '__main__':
    main(sys.argv)


