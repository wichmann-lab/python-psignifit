# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 14:25:56 2015

calculation of a first guess for the parameters
function Seed=getSeed(data,options)
This function finds a seed for initial optimization by logistic
regression of the data
Additional parameter is the options.widthalpha which specifies the
scaling of the width by
width= psi^(-1)(1-alpha) - psi^(-1)(alpha)
where psi^(-1) is the inverse of the sigmoid function.

@author: root
"""
import numpy as np
def getSeed(data,options):
    
    ''' input parsing '''
    alpha0 = options['widthalpha']
    if options['logspace']:
        data[:,0] = np.log(data[:,0])
    
    x = np.array(data[:,0])
    y = np.array(data[:,1])/np.array(data[:,2])
    
    #lower and upper asymptote taken simply from min/max of the data
    l = 1-np.max(y)
    gamma = np.min(y)
    
    # rescale y
    y = (y-gamma)/(1-l-gamma)
    
    '''prevent 0 and 1 as bad input for the logit. This moves the data in 
    from the borders by .25 of a trial from up and .25 of a trial from the
    bottom'''
    factor = .25/np.array(data[:,2])
    y = factor + (1-2*factor)*y
    # logit transform
    y = np.log(y/(1-y))
    
    fit = np.polyfit(x,y,1)
    fit = fit[[1,0]]        # change order of entries
    
    #threshold at the zero of the linear fit
    alpha = -fit[0]/fit[1]
    
    # width of the function difference between x'es where the logistic is alpha
    # and where it is 1-alpha
    beta = (np.log((1-alpha0)/alpha0) -np.log(alpha0/(1-alpha0)))/fit[1]
    varscale = np.exp(-20)    
    
    Seed = np.array([[alpha],[beta],[l],[gamma],[varscale]])  
            
    return Seed

if __name__ == "__main__":
    import sys
    getSeed(sys.argv[1], sys.argv[2])
