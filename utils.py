# -*- coding: utf-8 -*-
"""
Utils class capsulating all custom made probabilistic functions
"""

from numpy import ndarray, array, log, log1p, exp, sqrt, tan, pi, nan, isnan, inf, zeros, shape, ravel, tile, logical_and, logical_or
from scipy.special import betainc,betaln,erfcinv, erfc

def my_norminv(p,mu,sigma):
    
    x0 = -sqrt(2)*erfcinv(2*p)
    x = sigma*x0 + mu
    
    return x

def my_normcdf(x,mu,sigma):
    z = (x-mu) /sigma
    p = .5*erfc(-z/sqrt(2))
    
    return p

    
def my_t1cdf(x):
    '''    
    cumulative distribution function of a t-dist. with 1 degree of freedom
    function p=my_t1cdf(x)
    input
          x = point
          output
          p = cumulative probability

    see also: tcdf 
    '''
    xsq=x*x;
    p = betainc(1/2, 1/2,1 / (1 + xsq)) / 2
    p[x>0]=1-p[x>0]
    
    return p    

def my_t1icdf(p):
    x = tan(pi * (p - 0.5));
    return x    

def my_betapdf(x,a,b):
    ''' this implements the betapdf with less input checks '''
    
    if type(x) is int or float:
        x = array(x)

    # Initialize y to zero.
    y = zeros(shape(x))

    if len(ravel(a)) == 1:
        a = tile(a,shape(x))

    if len(ravel(b)) == 1:
        b = tile(b,shape(x))

    # Special cases
    y[logical_and(a==1, x==0)] = b[logical_and(a==1 , x==0)]
    y[logical_and(b==1 , x==1)] = a[logical_and(b==1 , x==1)]
    y[logical_and(a<1 , x==0)] = inf
    y[logical_and(b<1 , x==1)] = inf

    # Return NaN for out of range parameters.
    y[a<=0] = nan
    y[b<=0] = nan
    y[logical_or(logical_or(isnan(a), isnan(b)), isnan(x))] = nan

    # Normal values
    k = logical_and(logical_and(a>0, b>0),logical_and(x>0 , x<1))
    a = a[k]
    b = b[k]
    x = x[k]

    # Compute logs
    smallx = x<0.1

    loga = (a-1)*log(x)

    logb = zeros(shape(x))
    logb[smallx] = (b[smallx]-1) * log1p(-x[smallx])
    logb[~smallx] = (b[~smallx]-1) * log(1-x[~smallx])

    y[k] = exp(loga+logb - betaln(a,b))

    return y
    
def fill_kwargs(kw_args, values):
    '''
    Fill the empty dictionary kw_args with the values given in values.
    values are assigned in the order alpha, beta, lambda, gamma, varscale.
    '''
    
    kw_args.clear()
    d = len(values)    
    for i in range(0,d):
        if i == 0:
            if type(values[0]) == ndarray:
                kw_args['alpha'] = values[0]
            else:
                kw_args['alpha'] = array([values[0]])
        if i == 1:
            if type(values[1]) == ndarray:
                kw_args['beta'] = values[1]
            else:
                kw_args['beta'] = array([values[1]])                
        if i == 2:
            if type(values[2]) == ndarray:
                kw_args['lambda'] = values[2]
            else:
                kw_args['lambda'] = array([values[2]])
        if i == 3:
            if type(values[3]) == ndarray:
                kw_args['gamma'] = values[3]
            else:
                kw_args['gamma'] = array([values[3]])
        if i == 4:
            if type(values[4]) == ndarray:
                kw_args['varscale'] = values[4]
            else:
                kw_args['varscale'] = array([values[4]])
        
def strToDim(string):
    """
    Finds the number corresponding to a dim/parameter given as a string. 
    """
    s = string.lower()
    if s in ['threshold','thresh','m','t','alpha', '0']:    return 0,'Threshold'
    elif s in  ['width','w','beta', '1']:                   return 1,'Width'
    elif s in ['lapse','lambda','lapserate','lapse rate','lapse-rate',
               'upper asymptote','l', '2']:                 return 2, r'$\lambda$'
    elif s in ['gamma','guess','guessrate','guess rate',
               'guess-rate','lower asymptote','g', '3']:    return 3, r'$\gamma$'
    elif s in ['sigma','std','s','eta','e', '4']:           return 4, r'$\eta$'
    
