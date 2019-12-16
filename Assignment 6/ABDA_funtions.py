# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:40:27 2019

@author: johohm
"""
import numpy as np
def calculate_hdi(samples,percent=0.95):
    sorted_samples = np.sort(samples,0)   #Sort samples
    N = samples.size                    #Get size
    ll = int(np.ceil(N*percent))        #Calculate the interval that consist the desired samples
    
    dist_save = np.inf                  #Initiate the min distance
    for index,sample in enumerate(sorted_samples):  #Loop over the sorted samples
        if index + ll > N-1:                        #If start + interval is outside the array, break
            break
        end_sample = sorted_samples[index + ll]     #Otherwise calculate the distance between the start and end-point
        dist = end_sample-sample
        if dist < dist_save:                        #If it is shorter the previous save this as the interval 
            dist_save = dist
            l = sample
            u = end_sample
            
    return l,u                                      #Return the final values

def calculate_ci(samples,percent = 0.95):
    sorted_samples = np.sort(samples,0)       #Sort
    N = samples.size
    index =int( np.floor(N*(1-percent)/2))  #find the samples that corresponds to (1-percent)/2 lowest and highest values
    return sorted_samples[index],sorted_samples[N-1-index]

def calculate_mode(p):	#Not my code, found it on internet, computes mode from kde plot
	import scipy
	import numpy as np
	x,y = p.get_lines()[0].get_data()

	#care with the order, it is first y
	#initial fills a 0 so the result has same length than x
	#cdf = scipy.integrate.cumtrapz(y, x, initial=0)

	#nearest_05 = np.abs(cdf-0.5).argmin()
	
	
	
	idx = y.argmax()
	x_mode = x[idx]
	y_mode = y[idx]
	return x_mode,y_mode