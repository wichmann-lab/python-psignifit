# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as sp

def likelihood(data, options, args):
    """
    Calculates the (normalized) likelihood for the data from given parameters
    function [p,logPmax] = likelihood(typeHandle,data,alpha,beta,lambda,gamma)
    This function computes the likelihood for specific parameter values from
    the log-Likelihood
    The result is normalized to have maximum=1 because the Likelihoods become
    very small and this way stay in the range representable in floats

    """
    
    p = logLikelihood(data, options, args)
        
    '''We never need the actual value of the likelihood. Something proportional
    is enough and this circumvents numerical problems for the likelihood to
    become exactly 0'''
    
    logPmax = np.max(p)    
    
    p = p -np.max(p)
    p = np.exp(p)
    
    return (p,logPmax)


def logLikelihood(data,options, args):
    """
    The core function to evaluate the logLikelihood of the data
    function p=logLikelihood(data,options,alpha,beta,lambda,gamma,varscale)
    Calculates the logLikelihood of the given data with given parameter
    values. It is fully vectorized and contains the core calculations of
    psignifit.
    this actually adds the log priors as well. Technically it calculates the
    unnormalized log-posterior
    
    """

    
    sigmoidHandle = options['sigmoidHandle']
    if len(args) < 2:
        raise ValueError('not enough input parameters')
    else:
        if hasattr(args[0], '__iter__'):
            alpha = args[0]
        else:
            alpha = np.array([args[0]])
        if hasattr(args[1], '__iter__'):
            beta = args[1]        
        else:
            beta = np.array([args[1]])
            
        if len(args) > 2:
            if hasattr(args[2], '__iter__'):
                lamb = args[2]
            else:
                lamb = np.array([args[2]])
        else:
            lamb = np.array([0])
        if len(args) > 3:
            if hasattr(args[3], '__iter__'):
                gamma = args[3]
            else:
                gamma = np.array([args[3]])
        else:
            gamma = np.array([0.5])
        if len(args) > 4:
            if hasattr(args[4], '__iter__'):
                varscale = args[4]
            else:
                varscale = np.array([args[4]])
        else:
            varscale = np.array([1])
    
    # is the input only one point?
    oneParameter = not(len(alpha) > 1 or len(beta) > 1 or len(lamb) > 1 
                or len(gamma) > 1 or len(varscale) > 1)
    
    if oneParameter:     # in optimization if the parameter supplied is not the fixed value
        if np.isfinite(np.array(options['fixedPars'][0])):
            alpha = options['fixedPars'][0]
        if np.isfinite(np.array(options['fixedPars'][1])):
            beta = options['fixedPars'][1]
        if np.isfinite(np.array(options['fixedPars'][2])):
            lamb = options['fixedPars'][2]
        if np.isfinite(np.array(options['fixedPars'][3])):
            gamma = options['fixedPars'][3]
        if np.isfinite(np.array(options['fixedPars'][4])):
            varscale = options['fixedPars'][4]
        if (lamb<0) or (lamb>(1-gamma)):
            lamb = np.nan
        if (gamma<0) or (gamma>(1-lamb)):
            gamma = np.nan
        if (varscale<0) or (varscale>1):
            varscale = np.nan
    else:        
        #issues for automization: limit range for lambda & gamma
        lamb[np.logical_or(lamb < 0 ,lamb > (1-np.max(gamma)))] = np.nan
        gamma[np.logical_or(gamma < 0, gamma > (1-np.max(lamb)))] = np.nan
        varscale[np.logical_or(varscale < 0, varscale > 1)] = np.nan
    
    varscaleOrig = np.reshape(varscale, [1,1,1,1,-1]);

    
    if oneParameter:
        if options['expType'] == 'equalAsymptote':
            gamma = lamb
        p = 0
        scale = 1-gamma -lamb
        psi = np.array([sigmoidHandle(x,alpha, beta) for x in data[:,0]])
        psi = gamma + scale*psi
        psi = np.squeeze(psi)
        n = np.array(data[:,2])
        k = np.array(data[:,1])
        varscale = varscale**2;
        
        if varscale < 10**-9:
            p = p + k * np.log(psi) + (n-k)*np.log(1-psi)   # binomial model
        else:
            v = 1/varscale - 1
            a = psi*v                                       # alpha for binomial
            b = (1-psi)*v                                   # beta for binomial
            p = p + sp.gammaln(k+a) + sp.gammaln(n-k+b)
            p = p -sp.gammaln(n+v) - sp.gammaln(a) - sp.gammaln(b)
            p = p + sp.gammaln(v)
        p = np.sum(p)  # add up log likelihood
        if np.isnan(p):
            p = - np.inf
    else:       # for grid evaluation
        
        alpha = np.reshape(alpha, [-1,1,1,1,1])
        beta = np.reshape(beta, [1, -1, 1, 1,1])
        lamb = np.reshape(lamb, [1,1,-1, 1, 1])
        gamma = np.reshape(gamma, [1,1,1,-1, 1])
        varscale = np.reshape(varscale,[1,1,1,1,-1])
        varscale = varscale**2          # go from sd to variance
        vbinom = (varscale < 10**-9)    # for very small variance use the binomial model
        
        v = varscale[~vbinom]
        v = 1/v -1
        v = np.reshape(v, [1,1,1,1, -1])
        p = np.zeros((1,1,1,1,1))       # posterior
        pbin = np.zeros((1,1,1,1,1))    # posterior for binomial work
        n = np.size(data,0)
        levels = np.array(data[:,0])    # needed for GPU work
    
        
        if options['expType'] == 'equalAsymptote':
            gamma = lamb
        
        scale = 1-gamma-lamb
        for i in range(0,n):
            if options['verbose'] > 3: 
                print('\r%d/%d', i,n)
            xi = levels[i]
            psi = sigmoidHandle(xi,alpha,beta) 
            psi = psi*scale + gamma
            ni = np.array(data[i,2])
            ki = np.array(data[i,1])
            
            if ((ni-ki)>0 and ki > 0):
                psi[psi < 1E-200] = 1E-200
                temp = 1-psi
                temp[temp < 1E-200] = 1E-200
                
                pbin = pbin + ki * np.log(psi) + (ni-ki)*np.log(temp)
                if (v.size != 0):
                    a = psi * v
                    b = (1-psi)*v
                    p = p + sp.gammaln(ki+a) + sp.gammaln(ni-ki+b)
                    p = p - sp.gammaln(ni+v) - sp.gammaln(a) - sp.gammaln(b)
                    p = p +sp.gammaln(v)
                else:
                    p = np.array([])
            elif ki > 0:    # --> ni-ki == 0
                psi[psi < 1E-200] = 1E-200
                pbin  = pbin + ki * np.log(psi);
                if (v.size != 0):                                             
                    a = psi*v
                    p = p + sp.gammaln(ki + a)
                    p = p - sp.gammaln(ni+v)
                    p = p - sp.gammaln(a)
                    p = p + sp.gammaln(v)
                else:
                    p = np.array([])
            
            elif (ni-ki) > 0 :  # --> ki ==0
                pbin = pbin  + (ni-ki)*np.log(1-psi)
                if (v.size != 0):
                    b = (1-psi)*v
                    p = p + sp.gammaln(ni-ki+b)
                    p = p - sp.gammaln(ni+v) - sp.gammaln(b)
                    p = p + sp.gammaln(v)
                else:
                    p = np.array([])
        
        if options['verbose'] > 3 :
            print('\n')
        if (p.size == 0):
            p = np.tile(pbin, [1,1,1,1,np.sum(vbinom)])
        else:
            p = np.concatenate((np.tile(pbin, [1,1,1,1,np.sum(vbinom)]),p), axis=4)
        p[np.isnan(p)] = -np.inf
      
    if (options['priors']):
        
        if isinstance(options['priors'], list):
            if hasattr(options['priors'][0], '__call__'):
                temp = options['priors'][0](alpha)
                if len(np.ravel(temp)) > 1: temp[temp < 1E-200] = 1E-200
                prior = np.log(temp)
                if not(oneParameter):
                    prior = np.tile(prior, (1,) + p.shape[1:])
                p += prior
            if hasattr(options['priors'][1], '__call__'):
                temp = options['priors'][1](beta)
                if len(np.ravel(temp)) > 1: temp[temp < 1E-200] = 1E-200         
                
                prior = np.log(temp)
                if not(oneParameter):
                    prior = np.tile(prior, (p.shape[0],1) +p.shape[2:])
                p += prior
            if hasattr(options['priors'][2], '__call__'):
                temp = options['priors'][2](lamb)
                if len(np.ravel(temp)) > 1: temp[temp < 1E-200] = 1E-200             
                
                prior = np.log(temp)
                if not(oneParameter):
                    prior = np.tile(prior,p.shape[0:2] + (1,) + p.shape[3:])
                p += prior
            if hasattr(options['priors'][3], '__call__'):
                temp = options['priors'][3](gamma)
                if len(np.ravel(temp)) > 1: temp[temp < 1E-200] = 1E-200   
                prior = np.log(temp) 
                if not(oneParameter):
                    if options['expType'] == 'equalAsymptote':
                        prior = np.tile(prior, p.shape[0:2] +(1,1,p.shape[4])) 
                    else:
                        prior = np.tile(prior, p.shape[0:3] +(1,p.shape[4])) 
                p += prior
            if hasattr(options['priors'][4], '__call__'):
                temp = options['priors'][4](varscaleOrig)
                if len(np.ravel(temp)) > 1: temp[temp < 1E-200] = 1E-200                
                
                prior = np.log(temp)
                if not(oneParameter):
                    prior = np.tile(prior, p.shape[0:-1]+(1,))
                else:
                    prior = np.squeeze(prior)
                p += prior

    return p  

        
if __name__ == "__main__":
    import sys
    likelihood(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])

    
        
