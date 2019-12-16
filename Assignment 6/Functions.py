# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:59:38 2019

All functions to assignment 5 is defined in this file

@author: johohm
"""
import time
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import norm as normal
import matplotlib.pyplot as plt
import sys


def slice_sample(x0, w, log_pdf, N = 1000, m = 1e9, printing = True): 
        # A direct implementation from this paper
        # https://projecteuclid.org/download/pdf_1/euclid.aos/1056562461
        #Jespers code
        D = len(x0)
        xs = np.zeros((N,D))
        lp = np.zeros(N)

        if len(w.shape)==2:
            w = np.diag(w)

        for i in range(N):
            l = 1*x0                
            r = 1*x0                
            x1 = 1*x0

            for d in np.random.permutation(D):            
                lu = np.log(np.random.rand())
                u1 = np.random.rand()
                v1 = np.random.rand()  
                
                if i == 0:
                    y = log_pdf(x0) + lu
                    evals = 1
                else:
                    y = log_pdf_x1 + lu

                l[d] = x0[d] - u1*w[d] 
                r[d] = l[d] + w[d]
                
                j = np.floor(m*v1)
                k = (m-1)-j        
                while y < log_pdf(l) and j>0:
                    l[d] -= w[d]
                    j -= 1

                while y < log_pdf(r) and k>0:
                    r[d] += w[d]
                    k -= 1
                    
                while True:
                    u2 = np.random.rand()  
                    x1[d] = l[d] + u2*(r[d]-l[d])

                    log_pdf_x1 = log_pdf(x1)
                    evals += 1                    
                    if y <= log_pdf_x1:
                        x0[d] = x1[d]
                        break
                    elif x1[d]<x0[d]:
                        l[d] = x1[d]
                    elif x1[d]>x0[d]:
                        r[d] = x1[d]
                    else: 
                        print(y)
                        print(log_pdf_x1)
                        print(x1)
                        print(x0)
                        raise RuntimeError('shrinkage error')
                        
                   
            xs[i] = x1
            lp[i] = log_pdf_x1
            if printing:
                if i % int(N/100.0) == 0: 
                    sys.stdout.write('%d '%((i*100)/N))

        return xs, lp