# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:08:32 2019

@author: johohm
"""
#       import some stuff
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ABDA_funtions import calculate_hdi, calculate_mode
import scipy

#       Load data from sampler
filename = "Assignment6_test"       #Filename to data
results = pickle.load( open( filename, "rb" ) )


#       Split up the loaded thata int meaningful parameters
theta_samp = results['theta']
sigma_samp = results['sigma']
tau_samp = results['tau']
mu_samp = results['mu']
phi_samp = results['phi']


#       Plot the phi distribution with mode and HDI
plt.figure(1)
plt.clf()
p = sns.distplot(phi_samp)
l_phi,u_phi = calculate_hdi(phi_samp)
phi_mode_x,phi_mode_y = calculate_mode(p)
plt.vlines(phi_mode_x,0,phi_mode_y)  
plt.hlines(0.4,l_phi,u_phi,linewidth=2,color = 'r')  


#       Set thet labels and save fig
plt.title(r'Difference between adults and children')
plt.xlabel(r'$\phi$')
plt.ylabel('pdf')
plt.legend(['HDI','Mode','Posterior'])
plt.savefig('Phi.png',dpi = 300)



plt.figure(12)
plt.clf()
p = sns.distplot(np.exp(phi_samp))
l_phi,u_phi = calculate_hdi(np.exp(phi_samp))
phi_mode_x,phi_mode_y = calculate_mode(p)
plt.vlines(phi_mode_x,0,phi_mode_y)  
plt.hlines(0.4,l_phi,u_phi,linewidth=2,color = 'r')  


#       Set thet labels and save fig
plt.title(r'Difference between adults and children')
plt.xlabel(r'$\exp(\phi)$')
plt.ylabel('pdf')
plt.legend(['HDI','Mode','Posterior'])
plt.savefig('Expected_time.png',dpi = 300)






#       PLot the tau parameter
plt.figure(2)
plt.clf()
p = sns.distplot(tau_samp)

l_tau,u_tau = calculate_hdi(tau_samp)
tau_mode_x,tau_mode_y = calculate_mode(p)

plt.vlines(tau_mode_x,0,tau_mode_y)  
plt.hlines(0.4,l_tau,u_tau,linewidth=2,color = 'r')  

plt.title(r'Assignment 6')
plt.xlabel(r'$\tau$')
plt.ylabel('pdf')
plt.legend(['HDI','Mode','Posterior'])
plt.savefig('tau_a6.png',dpi = 300)


#       Define a normal distribution
def Gaussian(mu,sigma,x):
    return 1/sigma/np.sqrt(2*np.pi) * np.exp(-(x-mu)**2 / (2*sigma**2))



#       Generatet the group 
data = np.linspace(5,7,100)
prior_adults = Gaussian(np.mean(mu_samp),np.mean(tau_samp),data)
prior_kids = Gaussian(np.mean(mu_samp) + np.mean(phi_samp),np.mean(tau_samp),data)
plt.figure(4)
plt.clf()
plt.plot(data,prior_adults,data,prior_kids)
plt.xlabel(r'$\theta$')
plt.title(r'Prior')
plt.ylabel('pdf')
plt.legend(['Adults', 'Children'])
plt.savefig('Priors.png',dpi = 300)


# Define the sample new person function

def sample_new(mu_raw,tau_raw,sigma_raw,phi_raw,N,bernull_prob):
    logy = np.zeros(N,)
    selected = np.random.randint(0,mu_raw.size,N)   #Create a vector of indices randomly choosed
    for i,sample_i in enumerate(selected):          #Loop over the selected samples
        if np.random.rand(1) < bernull_prob:
            theta = np.random.randn(1)[0]*tau_raw[sample_i] + mu_raw[sample_i] + phi_raw[sample_i]
        else:
            theta = np.random.randn(1)[0]*tau_raw[sample_i] + mu_raw[sample_i]  #Generate theta samples using mu and tau
        logy[i] = np.random.randn(1)[0]*sigma_raw[sample_i] + theta         #Use theta and sigma to generatet data
    
    return logy

        
kids_fraction = 9/35 # 9 childs and 34 in total


# Sample new data

y_adults = np.exp(sample_new(mu_samp,tau_samp,sigma_samp,phi_samp,5000,0))
y_kids = np.exp(sample_new(mu_samp,tau_samp,sigma_samp,phi_samp,5000,1))
y_mixed = np.exp(sample_new(mu_samp,tau_samp,sigma_samp,phi_samp,5000,kids_fraction))




plt.figure(5)
plt.clf()
sns.distplot(y_adults)
sns.distplot(y_kids)
sns.distplot(y_mixed)
#Todo: This handles do not work with my mode estimater, perhaps needs to be in separate figures

plt.figure(97)
p_adults = sns.distplot(y_adults)
plt.figure(98)
p_kids = sns.distplot(y_kids)
plt.figure(99)
p_mixed = sns.distplot(y_mixed)

 # Calculate HDI and modes. Note that the modes do not work.
plt.figure(5)
l_adults,u_adults = calculate_hdi(y_adults)
l_kids,u_kids = calculate_hdi(y_kids)
l_mixed,u_mixed = calculate_hdi(y_mixed)
adults_x,adults_y = calculate_mode(p_adults)
kids_x,kids_y = calculate_mode(p_kids)
mixed_x,mixed_y = calculate_mode(p_mixed)
plt.close(97)
plt.close(98)
plt.close(99)


plt.figure(5)
#Set axis and save file
plt.title('New Random Person')
plt.xlabel(r'Reaction time, [ms]')
plt.ylabel('pdf')
plt.legend(['Adult','Child','Mixed'])
plt.savefig('new_person',dpi = 300)





