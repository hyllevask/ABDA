# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:44:10 2019

@author: johohm
"""
import numpy as np
from Functions import slice_sample
import pickle


#Define the priors

def log_prior_sigma(sigma):
    return np.log(sigma > 0)        #Nice compact way of writing it, thx Jesper
def log_prior_mu(mu):
    return 1
def log_prior_tau(tau):
    return np.log(tau > 0)
def log_prior_theta(theta,mu,tau):
    return np.sum(-np.log(tau) - (theta-mu)**2/(2*tau**2))     #Do we rly need the sum?
    

#Define the Log-Likelihood
     
def log_likelihood(beta):    
    theta = beta[:J]
    mu = beta[J]
    sigma = beta[J+1]
    tau = beta[J+2]
    res = 0
    for i,data in enumerate(std_log_data):
        j = ind[i]-1
        #print(j)
        res += (-np.log(sigma) - (data-theta[j])**2/(2*sigma*2)) 
    return res

#Define the log posterior as the sum of the functions abouve

def logpost(beta):
    theta = beta[:J]
    mu = beta[J]
    sigma = beta[J+1]
    tau = beta[J+2]
    
    return log_likelihood(beta) + log_prior_theta(theta,mu,tau) + log_prior_mu(mu) + log_prior_sigma(sigma) + log_prior_tau(tau)

##########################  DATA ############################################

y = np.array([607, 583, 521, 494, 369, 782, 570, 678, 467, 620, 425, 395, 346, 361, 310,
     300, 382, 294, 315, 323, 421, 339, 398, 328, 335, 291, 329, 310, 294, 321,
     286, 349, 279, 268, 293, 310, 259, 241, 243, 272, 247, 275, 220, 245, 268,
     357, 273, 301, 322, 276, 401, 368, 149, 507, 411, 362, 358, 355, 362, 324,
     332, 268, 259, 274, 248, 254, 242, 286, 276, 237, 259, 251, 239, 247, 260,
     237, 206, 242, 361, 267, 245, 331, 357, 284, 263, 244, 317, 225, 254, 253,
     251, 314, 239, 248, 250, 200, 256, 233, 427, 391, 331, 395, 337, 392, 352,
     381, 330, 368, 381, 316, 335, 316, 302, 375, 361, 330, 351, 186, 221, 278,
     244, 218, 126, 269, 238, 194, 384, 154, 555, 387, 317, 365, 357, 390, 320,
     316, 297, 354, 266, 279, 327, 285, 258, 267, 226, 237, 264, 510, 490, 458,
     425, 522, 927, 555, 550, 516, 548, 560, 545, 633, 496, 498, 223, 222, 309,
     244, 207, 258, 255, 281, 258, 226, 257, 263, 266, 238, 249, 340, 247, 216,
     241, 239, 226, 273, 235, 251, 290, 473, 416, 451, 475, 406, 349, 401, 334,
     446, 401, 252, 266, 210, 228, 250, 265, 236, 289, 244, 327, 274, 223, 327,
     307, 338, 345, 381, 369, 445, 296, 303, 326, 321, 309, 307, 319, 288, 299,
     284, 278, 310, 282, 275, 372, 295, 306, 303, 285, 316, 294, 284, 324, 264,
     278, 369, 254, 306, 237, 439, 287, 285, 261, 299, 311, 265, 292, 282, 271,
     268, 270, 259, 269, 249, 261, 425, 291, 291, 441, 222, 347, 244, 232, 272,
     264, 190, 219, 317, 232, 256, 185, 210, 213, 202, 226, 250, 238, 252, 233,
     221, 220, 287, 267, 264, 273, 304, 294, 236, 200, 219, 276, 287, 365, 438,
     420, 396, 359, 405, 397, 383, 360, 387, 429, 358, 459, 371, 368, 452, 358, 371])

ind = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5, 
       5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 
       7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
       10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 
       11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 
       12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 
       13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 
       15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 
       18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
       20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 
       21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 
       23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 
       24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 
       25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 
       28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 
       30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 34, 34, 34, 34, 34, 
       34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34])

    
    
########################################################################################

log_data = np.log(y)
ldm = np.mean(log_data)     #Sample mean
ldstd = np.std(log_data)    #Sample STD
std_log_data = (log_data - ldm)/ldstd   #Standardize the data


J = max(ind)
jump = np.ones(J-1+4) # jump length in slice sampling, must be less than 1 in this example otherwise it always jumps out of 0<theta<1  
N = 15000
burn_in = 1000
beta0 = np.ones((J+3))      #Initial value
samples, id = slice_sample(beta0,jump,logpost,N)        #Do the sampling

#Get the raw-data from the sampler
theta_raw = samples[burn_in:-1,:J]
mu_raw = samples[burn_in:-1,J]
sigma_raw = samples[burn_in:-1,J + 1]
tau_raw = samples[burn_in:-1,J + 2]

#Save the data to file to use in the load_and_plot_data.py file
filename = 'MCMC_run2'
outfile = open(filename,'wb')
pickle.dump([theta_raw,mu_raw,sigma_raw,tau_raw,ldm,ldstd],outfile)
outfile.close()