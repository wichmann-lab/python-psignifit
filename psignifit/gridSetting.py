# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 18:47:05 2015

set for each dim the stepN Xvalues adaptively
function gridSetting(data,options)
This tries to get equal steps in cummulated likelihood in the slice
thorugh the Seed
it evaluates at GridSetEval points from Xborders(:,1) to Xborders(:,2)

@author: root
"""
from scipy.stats import beta as b
import numpy as np
import copy

from .likelihood import likelihood

def gridSetting(data,options,Seed):
    
    # Initialisierung
    d = np.size(options['borders'],0)
    X1D = []
    '''Equal steps in cumulative distribution'''
    
    if options['gridSetType'] == 'cumDist':
        Like1D = np.zeros([options['GridSetEval'], 1])
        for idx in range(d):
            if options['borders'][idx, 0] < options['borders'][idx,1]:
                X1D.append(np.zeros([1, options['stepN'][idx]]))
                local_N_eval = options['GridSetEval']
                while any(np.diff(X1D[idx]) == 0):
                    Xtest1D = np.linspace(options['borders'][idx,0], options['borders'][idx,1], local_N_eval)
                    alpha = Seed[0]
                    beta = Seed[1]
                    l = Seed[2]
                    gamma = Seed[3]
                    varscale = Seed[4]
                    
                    if idx == 1:
                        alpha = Xtest1D
                    elif idx == 2:
                        beta = Xtest1D
                    elif idx == 3:
                        l = Xtest1D
                    elif idx == 4:
                        gamma = Xtest1D
                    elif idx == 5:
                        varscale = Xtest1D
                    
                    Like1D = likelihood(data, options, [alpha, beta, l, gamma, varscale])
                    Like1D = Like1D + np.mean(Like1D)*options['UniformWeight']
                    Like1D = np.cumsum(Like1D)
                    Like1D = Like1D/max(Like1D)
                    wanted = np.linspace(0,1,options['stepN'][idx])
                    
                    for igrid in range(options['stepN'][idx]):
                        X1D[idx].append(copy.deepcopy(Xtest1D[Like1D >= wanted, 0, 'first'])) #TODO check
                        
                    local_N_eval = 10*local_N_eval
            else: 
                X1D.append(copy.deepcopy(options['borders'][idx,0]))
        
        ''' equal steps in cumulative  second derivative'''
    elif (options['gridSetType'] in ['2', '2ndDerivative']):
        Like1D = np.zeros([options['GridSetEval'], 1])
        
        for idx in range(d):
            if options['borders'][idx,0] < options['borders'][idx,1]:
                X1D.append(np.zeros([1,options['stepN'][idx]]))
                local_N_eval = options['GridSetEval']
                while any(np.diff(X1D[idx] == 0)):
                    
                    Xtest1D = np.linspace(options['borders'][idx,0], options['borders'][idx,1], local_N_eval)
                    alpha = Seed[0]
                    beta = Seed[1]
                    l = Seed[2]
                    gamma = Seed[3]
                    varscale = Seed[4]
                    
                    if idx == 1:
                        alpha = Xtest1D
                    elif idx == 2:
                        beta = Xtest1D
                    elif idx == 3:
                        l = Xtest1D
                    elif idx == 4:
                        gamma = Xtest1D
                    elif idx == 5:
                        varscale = Xtest1D
                        
                    # calc likelihood on the line                        
                    Like1D = likelihood(data, options, [alpha, beta, l, gamma, varscale])
                    Like1D = np.abs(np.convolve(np.squeeze(Like1D), np.array([1,-2,1]), mode='same'))
                    Like1D = Like1D + np.mean(Like1D)*options['UniformWeight']
                    Like1D = np.cumsum(Like1D)
                    Like1D = Like1D/max(Like1D)
                    wanted = np.linspace(0,1,options['stepN'][idx])
        
                    for igrid in range(options['stepN'][idx]):
                        X1D[idx].append(copy.deepcopy(Xtest1D[Like1D >= wanted, 0, 'first'])) #ToDo
                    local_N_eval = 10*local_N_eval
                    
                    if local_N_eval > 10**7:
                        X1D[idx] = np.unique(np.array(X1D)) # TODO check
                        break
            else: 
                X1D.append(options['borders'][idx,0])
    
        ''' different choices for the varscale '''
        ''' We use STD now directly as parametrisation'''
    elif options['gridSetType'] in ['priorlike', 'STD', 'exp', '4power']:
        for i in range(4):
            if options['borders'](i,0) < options['borders'](i,1):
                X1D.append(np.linspace(options['borders'][i,0], options['borders'][i,1], options['stepN'][i]))
            else:
                X1D.append(copy.deepcopy(options['borders'][id,0]))
        if options['gridSetType'] == 'priorlike':
            maximum = b.cdf(options['borders'][4,1],1,options['betaPrior'])
            minimum = b.cdf(options['borders'][4,0],1,options['betaPrior'])
            X1D.append(b.ppf(np.linspace(minimum, maximum, options['stepN'][4]), 1, options['betaPrior']))
        elif options['gridSetType'] == 'STD':
            maximum = np.sqrt(options['borders'][4,1])
            minimum = np.sqrt(options['borders'][4,0])
            X1D.append((np.linspace(minimum, maximum, options['stepN'][4]))**2)
        elif options['gridSetType'] == 'exp':
            p = np.linspace(1,1,options['stepN'][4])
            X1D.append(np.log(p)/np.log(.1)*(options['borders'][4,1] - options['borders'][4,0]) + options['borders'][4,0])
        elif options['gridSetType'] == '4power':
            maximum = np.sqrt(options['borders'][4,1])
            minimum = np.sqrt(options['borders'][4,0])
            X1D.append((np.linspace(minimum, maximum, options['stepN'][4]))**4) 
        
        
    return X1D

if __name__ == "__main__":
    import sys
    gridSetting(sys.argv[1], sys.argv[2], sys.argv[3])
