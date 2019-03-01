# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 17:14:59 2016

 

@author: root
"""
import numpy as np
import warnings 

from .utils import my_norminv
from .getWeights import getWeights
from .likelihood import likelihood
from .marginalize import marginalize

def setBorders(data,options):
    """ 
    automatically set borders on the parameters based on were you sampled.
    function Borders=setBorders(data,options)
    this function  sets borders on the parameter values of a given function
    automaically

    It sets: -the threshold to be within the range of the data +/- 50%
              -the width to half the distance of two datapoints up to 10 times
                       the range of the data
              -the lapse rate to 0 to .5
              -the lower asymptote to 0 to .5 or fix to 1/n for nAFC
              -the varscale to the full range from almost 0 to almost 1
    """

    widthmin = options['widthmin']
    # lapse fix to 0 - .5    
    lapseB = np.array([0,.5])
    
    if options['expType'] == 'nAFC':
        gammaB = np.array([1/options['expN'], 1/options['expN']])
    elif options['expType'] == 'YesNo':
        gammaB = np.array([0, .5])
    elif options['expType'] == 'equalAsymptote':
        gammaB = np.array([np.nan, np.nan])
    
    # varscale from 0 to 1, 1 excluded!
    varscaleB = np.array([0, 1-np.exp(-20)])  
        
    # if range was not given take from data
    '''if options['stimulusRange'].size <= 1 :
        options['stimulusRange'] = np.array([min(data[:,0]), max(data[:,0])])
        stimRangeSet = False
    else:
        stimRangeSet= True
        if options['logspace']:
            options['stimulusRange'] = np.log(options['stimulusRange'])
    '''
    
    '''
     We then assume it is one of the reparameterized functions with
     alpha=threshold and beta= width
     The threshold is assumed to be within the range of the data +/-
     .5 times it's spread
    '''
    dataspread = options['stimulusRange'][1] - options['stimulusRange'][0]
    alphaB = np.array([options['stimulusRange'][0] - .5*dataspread, options['stimulusRange'][1] +.5*dataspread]).squeeze()

    ''' the width we assume to be between half the minimal distance of
    two points and 5 times the spread of the data 
    
    if len(np.unique(data[:,0])) > 1 and not(stimRangeSet):
        widthmin = np.min(np.diff(np.sort(np.unique(data[:,0]))))
    else :
        widthmin = 100*np.spacing(options['stimulusRange'][1])
        '''
    
    ''' We use the same prior as we previously used... e.g. we use the factor by
    which they differ for the cumulative normal function '''
    
    Cfactor = (my_norminv(.95,0,1) - my_norminv(.05, 0,1))/(my_norminv(1- options['widthalpha'], 0,1) - my_norminv(options['widthalpha'], 0,1))
    betaB  = np.array([widthmin, 3/Cfactor*dataspread])
    
    borders =[[alphaB], [betaB], [lapseB], [gammaB], [varscaleB]]
    borders = np.array(borders).squeeze()
    
    return borders 

def moveBorders(data,options):
    """
    move parameter-boundaries to save computing power 
    function borders=moveBorders(data, options)
    this function evaluates the likelihood on a much sparser, equally spaced
    grid definded by mbStepN and moves the borders in so that that 
    marginals below tol are taken away from the borders.
    
    this is meant to save computing power by not evaluating the likelihood in
    areas where it is practically 0 everywhere.
    """
    borders = []
    
    tol = options['maxBorderValue']
    d = options['borders'].shape[0]
    
    MBresult = {'X1D':[]}
    
    ''' move borders inwards '''
    for idx in range(0,d):
        if (len(options['mbStepN']) >= idx and options['mbStepN'][idx] >= 2 
            and options['borders'][idx,0] != options['borders'][idx,1]) :
            MBresult['X1D'].append(np.linspace(options['borders'][idx,0], options['borders'][idx,1], options['mbStepN'][idx]))
        else:
            if (options['borders'][idx,0] != options['borders'][idx,1] and options['expType'] != 'equalAsymptote'):
                warnings.warn('MoveBorders: You set only one evaluation for moving the borders!') 
            
            MBresult['X1D'].append( np.array([0.5*np.sum(options['borders'][idx])]))        
           
        
    MBresult['weight'] = getWeights(MBresult['X1D'])
    #kwargs = {'alpha': None, 'beta':None , 'lambda': None,'gamma':None , 'varscale':None }
    #fill_kwargs(kwargs,MBresult['X1D'])
    MBresult['Posterior'] = likelihood(data, options, MBresult['X1D'])[0] 
    integral = sum(np.reshape(MBresult['Posterior'], -1) * np.reshape(MBresult['weight'], -1))
    MBresult['Posterior'] /= integral

    borders = np.zeros([d,2])    
    
    for idx in range(0,d):
        (L1D,x,w) = marginalize(MBresult, np.array([idx]))
        if len(x.shape)>0:
            x1 = x[np.max([np.where(L1D*w >= tol)[0][0] - 1, 0])]
            x2 = x[np.min([np.where(L1D*w >= tol)[0][-1]+1, len(x)-1])]
        else:
            x1 = x
            x2 = x
        
        borders[idx,:] = [x1,x2]
    
    return borders


        

if __name__ == "__main__":
    import sys
    setBorders(sys.argv[1], sys.argv[2])
