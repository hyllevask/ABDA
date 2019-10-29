from __future__ import division
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pystan
from ABDA_funtions import calculate_hdi, calculate_ci


y = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1])  # coin flips
N = 20000       #Number of samples
ass4_dat = {'J': len(y),
            'y': y} #Format for stan


model = """
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
#Compile model
sm = pystan.StanModel(model_code=model)

# Train the model and generate samples for coin 1
fit = sm.sampling(data=ass4_dat,iter=N, chains=1)
la = fit.extract(permuted=True)  # return a dictionary of arrays
result_coin1 = la['theta']

#Calculate HDI and equally tailed ci
hdi_l1,hdi_u1 = calculate_hdi(result_coin1,0.95)
ci_l1,ci_u1 = calculate_ci(result_coin1,0.95)

#Print p(head)
print('P(head) 0.95 HDI: [%.3f,%.3f]' %(hdi_l1,hdi_u1) )
print('P(head) 0.95 Equally tailed CI: [%.3f,%.3f]' %(ci_l1,ci_u1) )

#Plot Everything
plt.figure(1)
plt.clf()
sns.distplot(result_coin1, hist=False, norm_hist=True, color="b", kde_kws={"shade": True})
plt.vlines(hdi_l1,0,1)
plt.vlines(hdi_u1,0,1)

plt.vlines(ci_l1,0,1,color='g')
plt.vlines(ci_u1,0,1,color='g')
plt.xlabel(r'$\theta$')
plt.ylabel('Posterior pdf')
plt.title('Coin 1')
plt.savefig('Coin1',dpi=300)
#plt.show()

#Calculate and print probability for P(theta > 0.5)
prob05 = np.sum(0.5 < result_coin1)/np.size(result_coin1)
print('Coin 1: P(theta > 0.5) = %.2f' %prob05)


#Coin 2
#New Data
y = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0])  # coin flips
ass4_dat2 = {'J': len(y),
            'y': y}

#No need to compile a new model, just sample new data given the observations
fit = sm.sampling(data=ass4_dat2,iter=N, chains=1)
la = fit.extract(permuted=True)  # return a dictionary of arrays
result_coin2 = la['theta']


#Calculate P(theta1 > theta2)
prob = np.sum(result_coin2 < result_coin1)/np.size(result_coin1)
print('P(theta1 > theta2) = %.2f' %prob )



# Create the difference data
dtheta = result_coin1-result_coin2
hdi_l2,hdi_u2 = calculate_hdi(dtheta,0.95)#Calculate HDI
print('dtheta 0.95 HDI: [%.3f,%.3f]' %(hdi_l2,hdi_u2) )




#Plot Coin 1 and Coin 2 in same plot + Plot dtheta
plt.figure(2)
plt.clf()
plt.subplot(2,1,1)
sns.distplot(result_coin1, hist=False, norm_hist=True, color="b", kde_kws={"shade": True})
sns.distplot(result_coin2, hist=False, norm_hist=True, color="r", kde_kws={"shade": True})
plt.xlabel(r'$\theta$')
plt.ylabel('Posterior')
plt.title('Coin 1 and Coin 2')
plt.legend(['Coin 1', 'Coin 2'])
subfig = plt.subplot(2,1,2)
sns.distplot(dtheta, hist=False, norm_hist=True, color="g", kde_kws={"shade": True})
plt.vlines(hdi_l2,0,1)
plt.vlines(hdi_u2,0,1)
plt.xlabel(r'$d\theta$')
plt.ylabel('Posterior')
plt.subplots_adjust(hspace = 0.5,wspace = 1)
plt.savefig('Coin1andCoin2',dpi = 300)
#plt.show()




