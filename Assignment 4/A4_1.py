
from __future__ import division
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pystan


# Recreate fig 6.4
# Functions pasted from last assignment
def beta(a,b,theta):
    #Theta needs to be equally spaced
    d_theta = theta[1] - theta[0]
    B = np.sum(theta**(a-1)*(1-theta)**(b-1)*d_theta)
    return theta**(a-1) * (1 - theta)**(b-1)/B

def log_likelihood(y,theta):
    # Theta needs to be a np.ndarry even if its single value

    out = np.zeros_like(theta)
    for i,th in enumerate(theta):
        out[i] = np.sum(log_bernulli(y,th))
    return out

def log_bernulli(y,theta):
    #Return Bernulli pmf for outcome y given parameter theta
    #return np.log((theta**y)*(1.0-theta)**(1.0-y))
    return y*np.log(theta) + (1-y)*np.log(1-theta)


#Create the data and store it for pystan
y = np.zeros(20, dtype=int)
y[0:16] = 1

ass4_dat = {'J': len(y),
            'y': y}


#Define the three different models with the three differnt priors

model_1 = """
data {
    int<lower=0> J; // number of flips
    int<lower=0,upper=1> y[J]; // coin flips
}
parameters {
    real<lower=0,upper=1> theta; // prob of getting a head 
}
transformed parameters {
// no transformed variables to use
}
model {
    theta ~ beta(250, 250); // prior distribution for theta
    y ~ bernoulli(theta); // likelihood, note that stan will create the posterior automatically. 
}
"""
model_2 = """
data {
    int<lower=0> J; // number of flips
    int<lower=0,upper=1> y[J]; // coin flips
}
parameters {
    real<lower=0,upper=1> theta; // prob of getting a head 
}
transformed parameters {
// no transformed variables to use
}
model {
    theta ~ beta(18.25, 6.75); // prior distribution for theta
    y ~ bernoulli(theta); // likelihood, note that stan will create the posterior automatically. 
}
"""
model_3 = """
data {
    int<lower=0> J; // number of flips
    int<lower=0,upper=1> y[J]; // coin flips
}
parameters {
    real<lower=0,upper=1> theta; // prob of getting a head 
}
transformed parameters {
// no transformed variables to use
}
model {
    theta ~ beta(1, 1); // prior distribution for theta
    y ~ bernoulli(theta); // likelihood, note that stan will create the posterior automatically. 
}
"""
#Stack the models
model_list = [model_1,model_2,model_3]
result = []

#Loop over the three different models
for index,model in enumerate(model_list):
    print('Model %i' %(index+1))
    #Compile Stan model
    sm = pystan.StanModel(model_code=model)

    # Run the model and generate samples
    fit = sm.sampling(data=ass4_dat,iter=10000, chains=1)
    la = fit.extract(permuted=True)  # return a dictionary of arrays
    result.append(la['theta'])


############# Plot Everything! ####################
theta = np.linspace(0,1,201)
plt.figure(1)
index = [1,2,3]
for i in index:
    # The different priors
    if i==1:
        a,b = 250,250
    elif i==2:
        a,b = 18.25,6.75
    else:
        a,b = 1,1
    
    #Create and plt prior
    bb = beta(a, b, theta)
    plt.subplot(3,3,i)
    plt.plot(theta,bb)
    plt.ylabel('Priori')
    plt.title('a = %.1d, b = %.1d' %(a,b))

    #Create and plt prior
    plt.subplot(3,3,i+3)
    plt.plot(theta,np.exp(log_likelihood(y,theta)))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.ylabel('Likelihood')


    #Plot the posterior result from stan
    plt.subplot(3,3,i+6)
    sns.distplot(result[i-1], hist=False, norm_hist=True, color="b", kde_kws={"shade": True})
    plt.xlim([0,1])
    plt.xlabel(r'$\theta$')
    plt.ylabel('Posterior')
    
#Make the spacing ok, then save it and finally display it    
plt.subplots_adjust(hspace = 0.8,wspace = 1.3)
plt.savefig('Fig64_stan',dpi=300)
#plt.show()
###################################################