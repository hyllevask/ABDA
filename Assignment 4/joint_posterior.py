# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:14:37 2019

@author: johohm
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pystan
from ABDA_funtions import calculate_hdi, calculate_ci




y1 = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1])  # coin flips
y2 = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0])  # coin flips
N = 50000       #Number of samples




ass4_dat = {'J1': len(y1),
            'J2':len(y2),
            'y1': y1,
            'y2': y2} #Format for stan

model = """
data {
    int<lower=0> J1; // number of flips
    int<lower=0> J2; // number of flips
    int<lower=0,upper=1> y1[J1]; // coin flips
    int<lower=0,upper=1> y2[J2]; // coin flips
}
parameters {
    real<lower=0,upper=1> theta1; // prob of getting a head 
    real<lower=0,upper=1> theta2; // prob of getting a head 
}
transformed parameters {
// no transformed variables to use
}
model {
    theta1 ~ beta(1, 1); // prior distribution for theta
    y1 ~ bernoulli(theta1); // likelihood, note that stan will create the posterior automatically.
    theta2 ~ beta(1, 1); // prior distribution for theta
    y2 ~ bernoulli(theta2); // likelihood, note that stan will create the posterior automatically. 
}
"""

sm = pystan.StanModel(model_code=model)

# Train the model and generate samples for coin 1
fit = sm.sampling(data=ass4_dat,iter=N, chains=1)
la = fit.extract(permuted=True)  # return a dictionary of arrays

sns.jointplot(la['theta1'],la['theta2'], kind="hex",xlim=(0,1),ylim=(0,1))