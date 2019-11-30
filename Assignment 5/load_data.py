import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ABDA_funtions import calculate_hdi, calculate_mode
import scipy


#Load the data from file
[theta_raw,mu_raw,sigma_raw,tau_raw,ldm,ldstd] = pickle.load( open( "MCMC_run3", "rb" ) )
sample_mean = pickle.load( open( "sample_mean_data", "rb" ) )


#Define function for sampling new group member
def sample_new(mu_raw,tau_raw,sigma_raw,N):
    logy = np.zeros(N,)
    selected = np.random.randint(0,mu_raw.size,N)   #Create a vector of indices randomly choosed
    for i,sample_i in enumerate(selected):          #Loop over the selected samples
        theta = np.random.randn(1)[0]*tau_raw[sample_i] + mu_raw[sample_i]  #Generate theta samples using mu and tau
        logy[i] = np.random.randn(1)[0]*sigma_raw[sample_i] + theta         #Use theta and sigma to generatet data
    
    return logy



# Un-standardize the data
theta_samp = theta_raw*ldstd + ldm
mu_samp = mu_raw*ldstd + ldm
sigma_samp = sigma_raw*ldstd
tau_samp = tau_raw*ldstd

#Restructure the data
sigma_samp = sigma_samp.reshape(sigma_samp.size,1)
mu_samp = mu_samp.reshape(mu_samp.size,1)
tau_samp = tau_samp.reshape(tau_samp.size,1)

#Calculate the expextency value of the LogNormal distribution
Ey_ind = np.exp(theta_samp + 0.5*sigma_samp **2)
Ey_group = np.exp(mu_samp + 0.5*sigma_samp**2 + 0.5*tau_samp**2)


########## The Dude ########################
l_dude,u_dude = calculate_hdi(Ey_ind[:,3])      #Calculate HDI
plt.figure(1)
plt.clf()
p = sns.distplot(Ey_ind[:,3],bins = 41)
plt.hlines(0.001,l_dude,u_dude,linewidth=2,color = 'r')     #Plot the HDI      
mode_x,mode_y = calculate_mode(p)                           #Calculate the mode from the KDE plot, good for low number of samples
plt.vlines(mode_x,0,mode_y)                                 #Plot the mode

#Set axis and save file
plt.title('The Dude')
plt.xlabel(r'$E[y]$, [ms]')
plt.ylabel('pdf')
plt.legend(['HDI','Mode','Posterior'])
plt.savefig('The_dude.png',dpi = 300)



############ Group Level ##################
l_group,u_group = calculate_hdi(Ey_group[:,0])
plt.figure(2)
plt.clf()
p2 = sns.distplot(Ey_group,bins = 41)
plt.hlines(0.001,l_group,u_group,linewidth=2,color = 'r')
mode_x_group,mode_y_group = calculate_mode(p2)
plt.vlines(mode_x_group,0,mode_y_group)


#Set axis and save file
plt.title('Group')
plt.xlabel(r'$E[y]$, [ms]')
plt.ylabel('pdf')
plt.legend(['HDI','Mode','Posterior'])
plt.savefig('Group.png',dpi = 300)


########### Sample Random New Member #########
#Get the new data and un-standardize it
y = np.exp(sample_new(mu_raw,tau_raw,sigma_raw,15000) * ldstd + ldm)

plt.figure(3)
plt.clf()
p3 = sns.distplot(y,bins = 41)
l_new,u_new = calculate_hdi(y)
plt.hlines(0.0005,l_new,u_new,linewidth=2,color = 'r')
mode_x_new,mode_y_new = calculate_mode(p3)
plt.vlines(mode_x_new,0,mode_y_new)

#Set axis and save file
plt.title('New Random Person')
plt.xlabel(r'Reaction time, [ms]')
plt.ylabel('pdf')
plt.legend(['HDI','Mode','Posterior'])
plt.savefig('Random_Person.png',dpi = 300)


#Compare with sample mean
plt.figure(4)
plt.clf()

#Use boxplot for the posterior and red markers for the sample mean
sns.boxplot(data = theta_samp,showfliers=False,color='lavender')
plt.plot(sample_mean,'ro',fillstyle = 'full')


#Set axis and save file
plt.xlabel('Person i')
plt.ylabel(r'$\theta_i$, [ms]')
plt.savefig('Box.png',dpi = 300)


















        
        
        
        
        
