# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 23:09:42 2015

@author = Wichmann Lab
translated by Sophie Laturnus

"""
import numpy as np
import warnings


from .utils import my_betapdf, my_norminv

def prior1(x, xspread, stimRange):
    
    r = (x >= (stimRange[0]-.5*xspread))*(x<=stimRange[0])*(.5+.5*np.cos(2*np.pi*(stimRange[0]-x)/xspread)) \
    + (x>stimRange[0])*(x<stimRange[1]) + (x>=stimRange[1])*(x<=stimRange[1]+.5*xspread)*(.5+.5*np.cos(2*np.pi*(x-stimRange[1])/xspread))
        
    return r

def prior2(x, Cfactor, wmin, wmax):
    
    r = ((x*Cfactor)>=wmin)*((x*Cfactor)<=2*wmin)*(.5-.5*np.cos(np.pi*((x*Cfactor)-wmin)/wmin)) \
        + ((x*Cfactor)>2*wmin)*((x*Cfactor)<wmax) \
        + ((x*Cfactor)>=wmax)*((x*Cfactor)<=3*wmax)*(.5+.5*np.cos(np.pi/2*(((x*Cfactor)-wmax)/wmax)))

    return r
    
def getStandardPriors(data, options):
    """sets the standard Priors
    function priors = getStandardPriors(data,options)
    The priors set here are the ones used if the user does supply own priors.
    Thus this functions constitutes a way to change the priors permanently
    note here that the priors here are not normalized. Psignifit takes care
    of the normalization implicitly. """


    priors = []    
    
    """ threshold """
    xspread = options['stimulusRange'][1]-options['stimulusRange'][0]
    ''' we assume the threshold is in the range of the data, for larger or
        smaller values we tapre down to 0 with a raised cosine across half the
        dataspread '''

    priors.append(lambda x: prior1(x,xspread,options['stimulusRange']))
    
    """width"""
    # minimum = minimal difference of two stimulus levels
    widthmin = options['widthmin']
    
    widthmax = xspread
    ''' We use the same prior as we previously used... e.g. we use the factor by
        which they differ for the cumulative normal function'''
    Cfactor = (my_norminv(.95,0,1) - my_norminv(.05,0,1))/( my_norminv(1-options['widthalpha'],0,1) - my_norminv(options['widthalpha'],0,1))
    
    priors.append(lambda x: prior2(x,Cfactor, widthmin, widthmax))
    
    """ asymptotes 
    set asymptote prior to the 1, 10 beta prior, which corresponds to the
    knowledge obtained from 9 correct trials at infinite stimulus level
    """
    
    priors.append(lambda x: my_betapdf(x,1,10))
    priors.append(lambda x: my_betapdf(x,1,10))
    
    """ sigma """
    be = options['betaPrior']
    priors.append(lambda x: my_betapdf(x,1,be))
    
    return priors
    
    
    def __call__(self):
        import sys
        
        return getStandardPriors(sys.argv[1], sys.argv[2])


def checkPriors(data,options):
    """
    this runs a short test whether the provided priors are functional
     function checkPriors(data,options)
     concretely the priors are evaluated for a 25 values on each dimension and
     a warning is issued for zeros and a error for nan and infs and negative
     values

    """


    if options['logspace'] :
        data[:,0] = np.log(data[:,0])
    
    """ on threshold 
    values chosen according to standard boarders 
    at the borders it may be 0 -> a little inwards """
    data_min = np.min(data[:,0])
    data_max = np.max(data[:,0])
    dataspread = data_max - data_min
    testValues = np.linspace(data_min - .4*dataspread, data_max + .4*dataspread, 25)
    
    testResult = options['priors'][0](testValues)

    testForWarnings(testResult, "the threshold")
    """ on width
    values according to standard priors
    """
    testValues = np.linspace(1.1*np.min(np.diff(np.sort(np.unique(data[:,0])))), 2.9*dataspread, 25)
    testResult = options['priors'][1](testValues)
    
    testForWarnings(testResult, "the width")
    
    """ on lambda
    values 0 to .9
    """
    testValues = np.linspace(0.0001,.9,25)
    testResult = options['priors'][2](testValues)
    
    testForWarnings(testResult, "lambda")
    
    """ on gamma
    values 0 to .9
    """
    testValues = np.linspace(0.0001,.9,25)
    testResult = options['priors'][3](testValues)
    
    testForWarnings(testResult, "gamma")
    
    """ on eta
    values 0 to .9
    """
    testValues = np.linspace(0,.9,25)
    testResult = options['priors'][4](testValues)
    
    testForWarnings(testResult, "eta")    
   
    
def testForWarnings(testResult, parameter):
    
    assert all(np.isfinite(testResult)), "the prior you provided for %s returns non-finite values" %parameter
    assert all(testResult >= 0), "the prior you provided for %s returns negative values" % parameter

    if any(testResult == 0):
        warnings.warn("the prior you provided for %s returns zeros" % parameter)

def normalizeFunction(func, integral):
        
    l = lambda x: func(x)/integral
    return l

def normalizePriors(options):
    """ 
    normalization of given priors
    function Priors=normalizePriors(options)
    This function normalizes the priors from the given options dict, to
    obtain normalized priors.
    This normalization makes later computations for the Bayesfactor and
    plotting of the prior easier.

     This should be run with the original borders to obtain the correct
     normalization
     
    """
    
    priors = []

    for idx in range(0,len(options['priors'])):
        if options['borders'][idx][1] > options['borders'][idx][0]:
            #choose xValues for calculation of the integral
            x = np.linspace(options['borders'][idx][0], options['borders'][idx][1], 1000)
            # evaluate unnormalized prior
            y = options['priors'][idx](x)
            w = np.convolve(np.diff(x), np.array([.5,.5]))
            integral = sum(y[:]*w[:])
            func = options['priors'][idx]
            priors.append(normalizeFunction(func,integral))
        else:
            priors.append(lambda x: np.ones_like(x,dtype='float'))
    
    return priors
    



if __name__ == "__main__":
    import sys
    getStandardPriors(sys.argv[1], sys.argv[2]) #TODO change accordingly?
    
