# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:40:27 2019

@author: johohm
"""
import numpy as np

def calculate_hdi(samples,percent=0.95):
    sorted_samples = np.sort(samples)
    N = samples.size
    ll = int(np.ceil(N*percent))
    
    dist_save = np.inf
    for index,sample in enumerate(sorted_samples):
        if index + ll > N-1:
            break
        end_sample = sorted_samples[index + ll]

        dist = end_sample-sample
        if dist < dist_save:
            dist_save = dist
            l = sample
            u = end_sample
            
    return l,u

def calculate_ci(samples,percent = 0.95):
    sorted_samples = np.sort(samples)
    N = samples.size
    index =int( np.floor(N*(1-percent)/2))
    return sorted_samples[index],sorted_samples[N-1-index]
    
    

#s = np.random.rand(1000)

#l,u = calculate_hdi(s,0.95)