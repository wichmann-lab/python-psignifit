# -*- coding: utf-8 -*-

import os as _os
import sys as _sys

import numpy as np
import datetime as _dt
import warnings
from copy import deepcopy as _deepcopy
import scipy

from . import likelihood as _l
from . import priors as _p
from . import borders as _b
from .utils import my_norminv as _my_norminv
from .utils import my_t1icdf as _my_t1icdf


from .gridSetting import gridSetting
from .getWeights import getWeights
from .getConfRegion import getConfRegion
from .getSeed import getSeed
from .marginalize import marginalize
from .poolData import poolData
from .getSigmoidHandle import getSigmoidHandle

from . import psigniplot as plot

def psignifit(data, optionsIn):
    """
    main function for fitting psychometric functions
    function result=psignifit(data,options)
    This function is the user interface for fitting psychometric functions to data.
        
    pass your data in the n x 3 matrix of the form:
    [x-value, number correct, number of trials]

    options should be a dictionary in which you set the options for your fit.
    You can find a full overview over the options in demo002
    
    The result of this function is a dictionary, which contains all information the 
    program produced for your fit. You can pass this as whole to all further 
    processing function provided with psignifit. Especially to the plot functions.
    You can find an explanation for all fields of the result in demo006
        
    To get an introduction to basic usage start with demo001
    """
    #--------------------------------------------------------------------------
    #input parsing
    #--------------------------------------------------------------------------
    # data
    data = np.array(data)
                # percent correct in data
    if all( np.logical_and(data[:,1] <= 1, data[:,1] >= 0)) and any(np.logical_and(data[:,1] > 0, data[:,1] < 1)):
        data[:,1] = round(map( lambda x, y: x*y, data[:,2],data[:,1])) # we try to convert into our notation
        
    # options
        
    if not('optionsIn' in locals()): 
        options = dict()
    else:
        options = _deepcopy(optionsIn)

    if not('sigmoidName' in options.keys()):
        options['sigmoidName'] = 'norm'
    
    if not('expType' in options.keys()):
        options['expType'] = 'YesNo'

    if not('estimateType' in options.keys()):
        options['estimateType'] = 'MAP'

    if not('confP' in options.keys()):
        options['confP'] = [.95, .9, .68]
        
    if not('instantPlot' in options.keys()):
        options['instantPlot'] = 0
        
    if not('maxBorderValue' in options.keys()):
        options['maxBorderValue'] = .00001
        
    if not('moveBorders' in options.keys()):
        options['moveBorders'] = 1
        
    if not('dynamicGrid' in options.keys()):
        options['dynamicGrid'] = 0
        
    if not('widthalpha' in options.keys()):
        options['widthalpha'] = .05
        
    if not('threshPC' in options.keys()):
        options['threshPC'] = .5

    if not('CImethod' in options.keys()):
        options['CImethod'] = 'percentiles'

    if not('gridSetType' in options.keys()):
        options['gridSetType'] = 'cumDist'
        
    if not( 'fixedPars' in options.keys()):
        options['fixedPars'] = np.ones(5)*np.nan
    elif len(options['fixedPars'].shape)>1:
        options['fixedPars'] = np.squeeze(options['fixedPars'])
    if not('nblocks' in options.keys()):
        options['nblocks'] = 25
    
    if not('useGPU' in options.keys()):
        options['useGPU'] = 0
    
    if not('poolMaxGap' in options.keys()):
        options['poolMaxGap'] = np.inf
    
    if not('poolMaxLength' in options.keys()):
        options['poolMaxLength'] = np.inf
    
    if not('poolxTol' in options.keys()):
        options['poolxTol'] = 0
    
    if not('betaPrior' in options.keys()):
        options['betaPrior'] = 10
    
    if not('verbose' in options.keys()):
        options['verbose'] = 0
        
    if not('stimulusRange' in options.keys()):
        options['stimulusRange'] = 0
        
    if not('fastOptim' in options.keys()):
        options['fastOptim'] = False
    
    if options['expType'] in ['2AFC', '3AFC', '4AFC']:            
        options['expN'] = int(float(options['expType'][0]))
        options['expType'] = 'nAFC'

    if options['expType'] == 'nAFC' and not('expN' in options.keys()):
        raise ValueError('For nAFC experiments please also pass the number of alternatives (options.expN)')
    
    if options['expType'] == 'YesNo':
        if not('stepN' in options.keys()):
            options['stepN'] = [40,40,20,20,20]
        if not('mbStepN' in options.keys()):
            options['mbStepN'] = [25,30, 10,10,15]
    elif options['expType'] == 'nAFC' or options['expType'] == 'equalAsymptote':
        if not('stepN' in options.keys()):
            options['stepN'] = [40,40,20,1,20]
        if not('mbStepN' in options.keys()):
            options['mbStepN'] = [30,40,10,1,20]
    else:
        raise ValueError('You specified an illegal experiment type')
    
    assert (max(data[:,0]) - min(data[:,0]) > 0), 'Your data does not have variance on the x-axis! This makes fitting impossible'
                 
                     
    '''
    log space sigmoids
    we fit these functions with a log transformed physical axis
    This is because it makes the paramterization easier and also the priors
    fit our expectations better then.
    The flag is needed for the setting of the parameter bounds in setBorders
    '''
    
    if options['sigmoidName'] in ['Weibull','logn','weibull']:
            options['logspace'] = 1
            assert min(data[:,0]) > 0, 'The sigmoid you specified is not defined for negative data points!'
    else:
        options['logspace'] = 0
        
    #if range was not given take from data
    if len(np.ravel(options['stimulusRange'])) <=1 :
        if options['logspace']:
            options['stimulusRange'] = np.array(np.log([min(data[:,0]),max(data[:,0])]))
        else :
            options['stimulusRange'] = np.array([min(data[:,0]),max(data[:,0])])

        stimRangeSet = False
    else:
        stimRangeSet = True
        if options['logspace']:
            options['stimulusRange'] = np.log(options['stimulusRange'])
    

    if not('widthmin' in options.keys()):
        if len(np.unique(data[:,0])) >1 and not(stimRangeSet):
            if options['logspace']:
                options['widthmin']  = min(np.diff(np.sort(np.unique(np.log(data[:,0])))))
            else:
                options['widthmin']  = min(np.diff(np.sort(np.unique(data[:,0]))))
        else:
            options['widthmin'] = 100*np.spacing(options['stimulusRange'][1])

    # add priors
    if options['threshPC'] != .5 and not(hasattr(options, 'priors')):
        warnings.warn('psignifit:TresholdPCchanged\n'\
            'You changed the percent correct corresponding to the threshold\n')    
    
    if not('priors' in options.keys()):
        options['priors'] = _p.getStandardPriors(data, options)
    else:
        
        priors = _p.getStandardPriors(data, options)
        
        for ipar in range(5):
            if not(hasattr(options['priors'][ipar], '__call__')):
                options['priors'][ipar] = priors[ipar]
                
        _p.checkPriors(data, options)
    if options['dynamicGrid'] and not('GridSetEval' in options.keys()):
        options['GridSetEval'] = 10000
    if options['dynamicGrid'] and not('UniformWeight' in options.keys()):
        options['UniformWeight'] = 1

    '''
    initialize
    '''        
    
    #warning if many blocks were measured
    if (len(np.unique(data[:,0])) >= 25) and (np.ravel(options['stimulusRange']).size == 1):
        warnings.warn('psignifit:probablyAdaptive\n'\
            'The data you supplied contained >= 25 stimulus levels.\n'\
            'Did you sample adaptively?\n'\
            'If so please specify a range which contains the whole psychometric function in options.stimulusRange.\n'\
            'This will allow psignifit to choose an appropriate prior.\n'\
            'For now we use the standard heuristic, assuming that the psychometric function is covered by the stimulus levels,\n'\
            'which is frequently invalid for adaptive procedures!')
    
    if all(data[:,2] <= 5) and (np.ravel(options['stimulusRange']).size == 1):
        warnings.warn('psignifit:probablyAdaptive\n'\
            'All provided data blocks contain <= 5 trials \n'\
            'Did you sample adaptively?\n'\
            'If so please specify a range which contains the whole psychometric function in options.stimulusRange.\n'\
            'This will allow psignifit to choose an appropriate prior.\n'\
            'For now we use the standard heuristic, assuming that the psychometric function is covered by the stimulus levels,\n'\
            'which is frequently invalid for adaptive procedures!')
    
    #pool data if necessary: more than options.nblocks blocks or only 1 trial per block
    if np.max(data[:,2]) == 1 or len(data) > options['nblocks']:
        warnings.warn('psignifit:pooling\n'\
            'We pooled your data, to avoid problems with n=1 blocks or to save time fitting because you have a lot of blocks\n'\
            'You can force acceptance of your blocks by increasing options.nblocks')
        data = poolData(data, options)
    
    
    # create function handle of sigmoid
    options['sigmoidHandle'] = getSigmoidHandle(options)
    
    # borders of integration
    if 'borders' in options.keys():
        borders = _b.setBorders(data, options)
        options['borders'][np.isnan(options['borders'])] = borders[np.isnan(options['borders'])]
    else:
        options['borders'] = _b.setBorders(data,options)
    
    border_idx = np.where(np.isnan(options['fixedPars']) == False);
    print(border_idx)
    if (border_idx[0].size > 0):
        options['borders'][border_idx[0],0] = options['fixedPars'][border_idx[0]]
        options['borders'][border_idx[0],1] = options['fixedPars'][border_idx[0]]
            
    # normalize priors to first choice of borders
    options['priors'] = _p.normalizePriors(options)
    if options['moveBorders']:
        options['borders'] = _b.moveBorders(data, options)
    
    ''' core '''
    result = psignifitCore(data,options)
        
    ''' after processing '''
    # check that the marginals go to nearly 0 at the borders of the grid
    if options['verbose'] > -5:
    
        if result['marginals'][0][0] * result['marginalsW'][0][0] > .001:
            warnings.warn('psignifit:borderWarning\n'\
                'The marginal for the threshold is not near 0 at the lower border.\n'\
                'This indicates that smaller Thresholds would be possible.')
        if result['marginals'][0][-1] * result['marginalsW'][0][-1] > .001:
            warnings.warn('psignifit:borderWarning\n'\
                'The marginal for the threshold is not near 0 at the upper border.\n'\
                'This indicates that your data is not sufficient to exclude much higher thresholds.\n'\
                'Refer to the paper or the manual for more info on this topic.')
        if result['marginals'][1][0] * result['marginalsW'][1][0] > .001:
            warnings.warn('psignifit:borderWarning\n'\
                'The marginal for the width is not near 0 at the lower border.\n'\
                'This indicates that your data is not sufficient to exclude much lower widths.\n'\
                'Refer to the paper or the manual for more info on this topic.')
        if result['marginals'][1][-1] * result['marginalsW'][1][-1] > .001:
            warnings.warn('psignifit:borderWarning\n'\
                'The marginal for the width is not near 0 at the lower border.\n'\
                'This indicates that your data is not sufficient to exclude much higher widths.\n'\
                'Refer to the paper or the manual for more info on this topic.')
    
    result['timestamp'] = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if options['instantPlot']:
        plot.plotPsych(result)
    
       
    
    return result
    
def psignifitFast(data,options):
    """
    this uses changed settings for the fit to obtain a fast point estimate to
    your data. 
    The mean estimate with these settings is very crude, the MAP estimate is
    better, but takes a bit of time for the optimization (~100 ms)
    """
    
    warnings.warn('You use the speed optimized version of this program. \n' \
    'This is NOT suitable for the final analysis, but meant for online analysis, adaptive methods etc. \n'  \
    'It has not been tested how good the estimates from this method are!')

    options['stepN']     = [20,20,10,10,1]
    options['mbStepN']  = [20,20,10,10,1]
    options['fixedPars'] = np.array([np.NaN,np.NaN,np.NaN,np.NaN,0.0])
    options['fastOptim'] = True
    
    res = psignifit(data,options)
    
    return res 
    

def psignifitCore(data, options):
    """
    This is the Core processing of psignifit, call the frontend psignifit!
    function result=psignifitCore(data,options)
    Data nx3 matrix with values [x, percCorrect, NTrials]

    sigmoid should be a handle to a function, which accepts
    X,parameters as inputs and gives back a value in [0,1]. ideally
    parameters(1) should correspond to the threshold and parameters(2) to
    the width (distance containing 95% of the function.
    """
    
    d = len(options['borders'])
    result = {'X1D': [], 'marginals': [], 'marginalsX': [], 'marginalsW': []}
    
    '''Choose grid dynamically from data'''
    if options['dynamicGrid']:
        # get seed from linear regression with logit transform
        Seed = getSeed(data,options)
        
        # further optimize the logliklihood to obtain a good estimate of the MAP
        if options['expType'] == 'YesNo':
            calcSeed = lambda X: -_l.logLikelihood(data, options, X[0], X[1], X[2], X[3], X[4])
            Seed = scipy.optimize.fmin(func=calcSeed, x0 = Seed)
        elif options['expType'] == 'nAFC':
            calcSeed = lambda X: -_l.logLikelihood(data, options, X[0], X[1], X[2], 1/options['expN'], X[3])
            Seed = scipy.optimize.fmin(func=calcSeed, x0 = [Seed[0:2], Seed[4]])
            Seed = [Seed[0:2], 1/options['expN'], Seed[3]] #ToDo check whether row or colum vector
        result['X1D'] = gridSetting(data,options, Seed) 
    
    
    else: # for types which do not need a MAP estimate
        if (options['gridSetType'] == 'priorlike' or options['gridSetType'] == 'STD'
            or options['gridSetType'] == 'exp' or options['gridSetType'] == '4power'):
                result['X1D'] = gridSetting(data,options) 
        else: # Use a linear grid
            for idx in range(0,d):
                # If there is an actual Interval
                if options['borders'][idx, 0] < options['borders'][idx,1]: 
                    
                    result['X1D'].append(np.linspace(options['borders'][idx,0], options['borders'][idx,1],
                                    num=options['stepN'][idx]))
                # if parameter was fixed
                else:
                    result['X1D'].append(np.array([options['borders'][idx,0]]))
                    
    '''Evaluate likelihood and form it into a posterior'''
    
    (result['Posterior'], result['logPmax']) = _l.likelihood(data, options, result['X1D'])
    result['weight'] = getWeights(result['X1D'])
    integral = np.sum(np.array(result['Posterior'][:])*np.array(result['weight'][:]))
    result['Posterior'] = result['Posterior']/integral
    result['integral'] = integral
    
    '''Compute marginal distributions'''
    
    for idx in range(0,d):
        m, mX, mW = marginalize(result,np.array(idx))
        result['marginals'].append(m)
        result['marginalsX'].append(mX)
        result['marginalsW'].append(mW) 
    
    result['marginals'] = np.squeeze(result['marginals'])
    result['marginalsX'] = np.squeeze(result['marginalsX'])
    result['marginalsW'] = np.squeeze(result['marginalsW'])
        
    '''Find point estimate'''
    if (options['estimateType'] in ['MAP','MLE']):
        # get MLE estimate
    
        #start at most likely grid point
        index = np.where(result['Posterior'] == np.max(result['Posterior'].ravel()))
      
        Fit = np.zeros([d,1])
        for idx in range(0,d):
            Fit[idx] = result['X1D'][idx][index[idx]] 
        
        if options['expType'] == 'YesNo':
            fun = lambda X, f: -_l.logLikelihood(data, options, [X[0],X[1],X[2],X[3],X[4]])
            x0 = _deepcopy(Fit)
            a = None
            
        elif options['expType'] == 'nAFC':
            #def func(X,f):
            #    return -_l.logLikelihood(data,options, [X[0], X[1], X[2], f, X[3]])
            #fun = func
            fun = lambda X, f:  -_l.logLikelihood(data,options, [X[0], X[1], X[2], f, X[3]])
            x0 = _deepcopy(Fit[0:3]) # Fit[3]  is excluded
            x0 = np.append(x0,_deepcopy(Fit[4]))
            a = np.array([1/options['expN']])
            
        elif options['expType'] == 'equalAsymptote':
            fun = lambda X, f: -_l.logLikelihood(data,options,[X[0], X[1], X[2], f, X[3]])
            x0 = _deepcopy(Fit[0:3])
            x0 = np.append(x0,_deepcopy(Fit[4]))
            a =  np.array([np.nan])
           
        else:
            raise ValueError('unknown expType')
            
        if options['fastOptim']:           
            Fit = scipy.optimize.fmin(fun, x0, args = (a,), xtol=0, ftol = 0, maxiter = 100, maxfun=100)
            warnings.warn('changed options for optimization')
        else:            
            Fit = scipy.optimize.fmin(fun, x0, args = (a,), disp = False)
        
        if options['expType'] == 'YesNo':
            result['Fit'] = _deepcopy(Fit)
        elif options['expType'] == 'nAFC': 
            fit = _deepcopy(Fit[0:3])
            fit = np.append(fit, np.array([1/options['expN']]))
            fit = np.append(fit, _deepcopy(Fit[3]))
            result['Fit'] = fit
        elif options['expType'] =='equalAsymptote':
            fit = _deepcopy(Fit[0:3])
            fit = np.append(fit, Fit[2])
            fit = np.append(fit, Fit[3])
            result['Fit'] = fit
        else:
            raise ValueError('unknown expType')
        
        par_idx = np.where(np.isnan(options['fixedPars']) == False)
        for idx in par_idx[0]:
            result['Fit'][idx] = options['fixedPars'][idx]
            
    elif options['estimateType'] == 'mean':
        # get mean estimate
        Fit = np.zeros([d,1])
        for idx in range[0:d]:
            Fit[idx] = np.sum(result['marginals'][idx]*result['marginalsW'][idx]*result['marginalsX'][idx])
        
        result['Fit'] = _deepcopy(Fit)
        Fit = np.empty(Fit.shape)
    '''Include input into result'''
    result['options'] = options # no copies here, because they are not changing
    result['data'] = data
    
    '''Compute confidence intervals'''
    if ~options['fastOptim']:
        result['conf_Intervals'] = getConfRegion(result)
        
    return result


def getSlope(result, stimLevel):
    """ 
    function slope = getSlope(result, stimLevel)
    This function finds the slope of the psychometric function at a given
    performance level in percent correct. 
    
    result is a result dictionary from psignifit
    
    stimLevel is the stimuluslevel at where to evaluate the slope
    
    This function cannot provide credible intervals. 
    """

    if 'Fit' in result.keys():
        theta0 = result['Fit']
    else:
        raise ValueError('Result needs to contain a resulting fit generated by psignifit')
        


    #calculate point estimate -> transform only the fit

    alpha = result['options']['widthalpha']
    if 'threshPC' in result['options'].keys():
        PC    = result['options']['threshPC']
    else:
        PC = 0.5
    
    if result['options']['sigmoidName'][0:3]=='neg':
        PC = 1-PC;


    sigName = result['options']['sigmoidName']
    
    if sigName in ['norm','gauss','neg_norm','neg_gauss']:
        C         = _my_norminv(1-alpha,0,1) - _my_norminv(alpha,0,1)
        normalizedStimLevel = (stimLevel-theta0[0])/theta0[1]*C
        slopeNormalized = scipy.stats.norm.pdf(normalizedStimLevel)
        slope = slopeNormalized *C/theta0[1]
    elif sigName in ['logistic','neg_logistic']:
        C = 2 * np.log(1/alpha - 1) / theta0[1]
        d = np.log(1/PC-1)
        slope = C*np.exp(-C*(stimLevel-theta0[0])+d)/(1+np.exp(-C*(stimLevel-theta0[0])+d))**2
    elif sigName in  ['gumbel','neg_gumbel']:
        C      = np.log(-np.log(alpha)) - np.log(-np.log(1-alpha))
        stimLevel = C/theta0[1]*(stimLevel-theta0[0])+np.log(-np.log(1-PC))
        slope = C/theta0[1]*np.exp(-np.exp(stimLevel))*np.exp(stimLevel)
    elif sigName in ['rgumbel','neg_rgumbel']:                  #reversed gumbel
        C      = np.log(-np.log(1-alpha)) - np.log(-np.log(alpha))
        stimLevel = C/theta0[1]*(stimLevel-theta0[0])+np.log(-np.log(PC))
        slope = -C/theta0[1]*np.exp(-np.exp(stimLevel))*np.exp(stimLevel)
    elif sigName in ['logn','neg_logn']:
        C      = _my_norminv(1-alpha,0,1) - _my_norminv(alpha,0,1)
        normalizedStimLevel = (np.log(stimLevel)-theta0[0])/theta0[1]
        slopeNormalized = scipy.stats.norm.pdf(normalizedStimLevel)
        slope = slopeNormalized *C/theta0[1]/stimLevel 
    elif sigName in ['Weibull','weibull','neg_Weibull','neg_weibull']:
        C      = np.log(-np.log(alpha)) - np.log(-np.log(1-alpha))
        stimLevelNormalized = C/theta0[1]*(np.log(stimLevel)-theta0[0])+np.log(-np.log(1-PC))
        slope = C/theta0[1]*np.exp(-np.exp(stimLevelNormalized))*np.exp(stimLevelNormalized)
        slope = slope/stimLevel
    elif sigName in ['tdist','student','heavytail','neg_tdist','neg_student','neg_heavytail']:
        # student T distribution with 1 df --> heavy tail distribution
        C      = (my_t1icdf(1-alpha) - my_t1icdf(alpha))
        stimLevel = (stimLevel-theta0[0])/theta0[1]*C+my_t1icdf(PC)
        slope = C/theta0[1]*t.pdf(stimLevel,df=1)
    else:
        raise ValueError('unknown sigmoid function')


    slope   = (1-theta0[2]-theta0[3])*slope
    
    if result['options']['sigmoidName'][0:3]=='neg':
        slope = -slope
    return slope
    

def getSlopePC(result, pCorrect, unscaled = False):
    """ 
    function slope = getSlopePC(result, pCorrect, unscaled = False)
    This function finds the slope of the psychometric function at a given
    performance level in percent correct. 
    
    result is a result dictionary from psignifit
    
    pCorrrect is the proportion correct at which to evaluate the slope
    
    This function cannot provide credible intervals. 
    """
    if 'Fit' in result.keys():
        theta0 = result['Fit']
    else:
        raise ValueError('Result needs to contain a resulting fit generated by psignifit')
        


    #calculate point estimate -> transform only the fit

    alpha = result['options']['widthalpha']
    if 'threshPC' in result['options'].keys():
        PC    = result['options']['threshPC']
    else:
        PC = 0.5

    if unscaled:
        assert ((pCorrect > 0) & (pCorrect < 1)), 'pCorrect must be in ]0,1[ '
        pCorrectUnscaled = pCorrect
    else:
        assert ((pCorrect > theta0[3]) & (pCorrect < (1-theta0[2]))), 'pCorrect must lay btw {:.2f} and {:.2f}'.format(theta0[3], (1-theta0[2]))
        pCorrectUnscaled = (pCorrect-theta0[3])/(1-theta0[2] - theta0[3])
        


    ''' find the (normalized) stimulus level, where the given percent correct is
    reached and evaluate slope there'''
    sigName = result['options']['sigmoidName'].lower()
    
    if sigName[0:3]=='neg':
        pCorrectUnscaled = 1-pCorrectUnscaled
        PC = 1-PC


    if sigName in ['norm', 'gauss','neg_norm', 'neg_gauss']:
        C         = _my_norminv(1-alpha,0,1) - _my_norminv(alpha,0,1)
        normalizedStimLevel = _my_norminv(pCorrectUnscaled,0,1)
        slopeNormalized = scipy.stats.norm.pdf(normalizedStimLevel)
        slope = slopeNormalized *C/theta0[1]
    elif sigName in ['logistic','neg_logistic']:
        stimLevel = theta0[0] - theta0[1]*np.log((1/pCorrectUnscaled-1)-np.log(1/PC-1))/2/np.log(1/alpha-1)
        C = 2 * np.log(1/alpha - 1) / theta0[1]
        d = np.log(1/PC-1)
        slope = C*np.exp(-C*(stimLevel-theta0[0])+d)/(1+np.exp(-C*(stimLevel-theta0[0])+d))**2
    elif sigName in  ['gumbel','neg_gumbel']:
        C      = np.log(-np.log(alpha)) - np.log(-np.log(1-alpha))
        stimLevel = np.log(-np.log(1-pCorrectUnscaled))
        slope = C/theta0[1]*np.exp(-np.exp(stimLevel))*np.exp(stimLevel)
    elif sigName in ['rgumbel','neg_rgumbel']:                  #reversed gumbel
        C      = np.log(-np.log(1-alpha)) - np.log(-np.log(alpha))
        stimLevel = np.log(-np.log(pCorrectUnscaled))
        slope = -C/theta0[1]*np.exp(-np.exp(stimLevel))*np.exp(stimLevel)       
    elif sigName in['logn', 'neg_logn']:
        C      = _my_norminv(1-alpha,0,1) - _my_norminv(alpha,0,1)
        stimLevel = np.exp(_my_norminv(pCorrectUnscaled, theta0[0]-_my_norminv(PC,0,theta0[1]/C), theta0[1] / C))
        normalizedStimLevel = _my_norminv(pCorrectUnscaled, 0,1)        
        slopeNormalized = scipy.stats.norm.pdf(normalizedStimLevel)
        slope = slopeNormalized *C/theta0[1]/stimLevel 
    elif sigName in ['Weibull','weibull','neg_Weibull','neg_weibull']:
        C      = np.log(-np.log(alpha)) - np.log(-np.log(1-alpha))
        stimLevel = np.exp(theta0[0]+theta0[1]/C*(np.log(-np.log(1-pCorrectUnscaled))-np.log(-np.log(1-PC))))
        stimLevelNormalized = np.log(-np.log(1-pCorrectUnscaled))
        slope = C/theta0[1]*np.exp(-np.exp(stimLevelNormalized))*np.exp(stimLevelNormalized)
        slope = slope/stimLevel
    elif sigName in ['tdist','student','heavytail','neg_tdist','neg_student','neg_heavytail']:
        # student T distribution with 1 df --> heavy tail distribution
        C      = (_my_t1icdf(1-alpha) - _my_t1icdf(alpha))
        stimLevel = _my_t1icdf(pCorrectUnscaled)
        slope = C/theta0[1]*t.pdf(stimLevel,df=1)
    else:
        raise ValueError('unknown sigmoid function')


    slope   = (1-theta0[2]-theta0[3])*slope
    
    if sigName[0:3]=='neg':
        slope = -slope
    return slope
    
    
def getThreshold(result,pCorrect, unscaled=False):
    """
    function [threshold,CI] = getThreshold(result, pCorrect,unscaled)
     this function finds a threshold value for a given fit for different
     percent correct cutoffs
    
     result is a result dict from psignifit
    
     pCorrect is the percent correct at the threshold you want to calculate
    
     unscaled is whether the percent correct you provide are for the unscaled
     sigmoid or for the one scaled by lambda and gamma. By default this
     function returns the one for the scaled sigmoid.
    
     The CIs you may obtain from this are calculated based on the confidence
     intervals only, e.g. with the shallowest and the steepest psychometric
     function and may thus broaden if you move away from the standard
     threshold of unscaled sigmoid = .5 /= options['threshPC']
    
     For the sigmoids in logspace this also returns values in the linear
     stimulus level domain.
    
    
     For a more accurate inference use the changed level of percent correct
     directly to define the threshold in the inference by setting
     options['threshPC'] and adjusting the priors.
    """

    if 'Fit' in result.keys():
        theta0 = _deepcopy(result['Fit'])
    else:
        raise ValueError('Result needs to contain a resulting fit generated by psignifit.')
    if 'conf_Intervals' in result.keys():
        CIs = _deepcopy(result['conf_Intervals'])
    else:
        raise ValueError('Result needs to contain confidence intervals for the fitted parameter.')
    

    if unscaled: # set asymptotes to 0 for everything.
        theta0[2]  = 0
        theta0[3]  = 0
        CIs[2:4,:] = 0
        
    
    assert ((np.array(pCorrect)>theta0[3]) & (np.array(pCorrect)<(1-theta0[2]))), 'The threshold percent correct is not reached by the sigmoid!'
        
    pCorrectUnscaled = (pCorrect-theta0[3])/(1-theta0[2]-theta0[3])
    alpha = result['options']['widthalpha']
    if  'threshPC' in result['options'].keys():
        PC    = result['options']['threshPC']
    else :
        PC = 0.5

    sigName = result['options']['sigmoidName'].lower()
    if sigName[0:3]=='neg':
        PC = 1-PC
        pCorrectUnscaled = 1-pCorrectUnscaled

    if sigName in ['norm', 'gauss','neg_norm', 'neg_gauss']:
        C         = _my_norminv(1-alpha,0,1) - _my_norminv(alpha,0,1)
        threshold = _my_norminv(pCorrectUnscaled, theta0[0]-_my_norminv(PC,0,theta0[1]/C), theta0[1] / C)
    elif sigName in ['logistic','neg_logistic']:
        threshold = theta0[0]-theta0[1]*(np.log(1/pCorrectUnscaled-1)-np.log(1/PC-1))/2/np.log(1/alpha-1)
    elif sigName in ['gumbel','neg_gumbel']:
        C      = np.log(-np.log(alpha)) - np.log(-np.log(1-alpha))
        threshold = theta0[0] + (np.log(-np.log(1-pCorrectUnscaled))-np.log(-np.log(1-PC)))*theta0[1]/C
    elif sigName in ['rgumbel','neg_rgumbel']:
        C      = np.log(-np.log(1-alpha)) - np.log(-np.log(alpha))
        threshold = theta0[0] + (np.log(-np.log(pCorrectUnscaled))-np.log(-np.log(PC)))*theta0[1]/C
    elif sigName in ['logn','neg_logn']:
        C      = _my_norminv(1-alpha,0,1) - _my_norminv(alpha,0,1)
        threshold = np.exp(_my_norminv(pCorrectUnscaled, theta0[0]-_my_norminv(PC,0,theta0[1]/C), theta0[1] / C))
    elif sigName in ['Weibull','weibull','neg_Weibull','neg_weibull']:
        C      = np.log(-np.log(alpha)) - np.log(-np.log(1-alpha))
        threshold = np.exp(theta0[0]+theta0[1]/C*(np.log(-np.log(1-pCorrectUnscaled))-np.log(-np.log(1-PC))))
    elif sigName in ['tdist','student','heavytail','neg_tdist','neg_student','neg_heavytail']:
        C      = (_my_t1icdf(1-alpha) - _my_t1icdf(alpha))
        threshold = (_my_t1icdf(pCorrectUnscaled)-_my_t1icdf(PC))*theta0[1] / C + theta0[0]
    else:
        raise ValueError('unknown sigmoid function')

    """ calculate CI -> worst case in parameter confidence intervals """
    
    warnings.warn('The CIs computed by this method are only upper bounds. For more accurate inference change threshPC in the options.')
    CI = np.zeros([len(result['options']['confP']),2])
    for iConfP in range(0,len(result['options']['confP'])):

        if sigName[0:3]=='neg':
            if pCorrectUnscaled < PC:
                thetaMax = [CIs[0,1,iConfP],CIs[1,0,iConfP],CIs[2,0,iConfP],CIs[3,1,iConfP],0]
                thetaMin = [CIs[0,0,iConfP],CIs[1,1,iConfP],CIs[2,1,iConfP],CIs[3,0,iConfP],0]
            else:
                thetaMin = [CIs[0,1,iConfP],CIs[1,1,iConfP],CIs[2,0,iConfP],CIs[3,1,iConfP],0]
                thetaMax = [CIs[0,0,iConfP],CIs[1,0,iConfP],CIs[2,1,iConfP],CIs[3,0,iConfP],0]
        else:
            if pCorrectUnscaled > PC:
                thetaMin = [CIs[0,0,iConfP],CIs[1,0,iConfP],CIs[2,0,iConfP],CIs[3,1,iConfP],0]
                thetaMax = [CIs[0,1,iConfP],CIs[1,1,iConfP],CIs[2,1,iConfP],CIs[3,0,iConfP],0]
            else:
                thetaMin = [CIs[0,0,iConfP],CIs[1,1,iConfP],CIs[2,0,iConfP],CIs[3,1,iConfP],0]
                thetaMax = [CIs[0,1,iConfP],CIs[1,0,iConfP],CIs[2,1,iConfP],CIs[3,0,iConfP],0]
        pCorrMin = (pCorrect-thetaMin[3])/(1-thetaMin[2]-thetaMin[3])
        pCorrMax = (pCorrect-thetaMax[3])/(1-thetaMax[2]-thetaMax[3])
        if sigName in ['norm', 'gauss','neg_norm', 'neg_gauss']:
            CI[iConfP,0]     = _my_norminv(pCorrMin, thetaMin[0]-_my_norminv(PC,0,thetaMin[1]/C), thetaMin[1] / C)
            CI[iConfP,1]     = _my_norminv(pCorrMax, thetaMax[0]-_my_norminv(PC,0,thetaMax[1]/C), thetaMax[1] / C)
        elif sigName in ['logistic','neg_logistic']:
            CI[iConfP,0]     = thetaMin[0]-thetaMin[1]*(np.log(1/pCorrMin-1)-np.log(1/PC-1))/2/np.log(1/alpha-1)
            CI[iConfP,1]     = thetaMax[0]-thetaMax[1]*(np.log(1/pCorrMax-1)-np.log(1/PC-1))/2/np.log(1/alpha-1)
        elif sigName in ['gumbel','neg_gumbel']:
            CI[iConfP,0] = thetaMin[0] + (np.log(-np.log(1-pCorrMin))-np.log(-np.log(1-PC)))*thetaMin[1]/C
            CI[iConfP,1] = thetaMax[0] + (np.log(-np.log(1-pCorrMax))-np.log(-np.log(1-PC)))*thetaMax[1]/C
        elif sigName in ['rgumbel','neg_rgumbel']:
            CI[iConfP,0] = thetaMin[0] + (np.log(-np.log(pCorrMin))-np.log(-np.log(PC)))*thetaMin[1]/C
            CI[iConfP,1] = thetaMax[0] + (np.log(-np.log(pCorrMax))-np.log(-np.log(PC)))*thetaMax[1]/C
        elif sigName in ['logn','neg_logn']:
            CI[iConfP,0] = np.exp(_my_norminv(pCorrMin, thetaMin[0]-_my_norminv(PC,0,thetaMin[1]/C), thetaMin[1]/ C))
            CI[iConfP,1] = np.exp(_my_norminv(pCorrMax, thetaMax[0]-_my_norminv(PC,0,thetaMax[1]/C), thetaMax[1] / C))
        elif sigName in ['Weibull','weibull','neg_Weibull','neg_weibull']:
            CI[iConfP,0] = np.exp(thetaMin[0]+thetaMin[1]/C*(np.log(-np.log(1-pCorrMin))-np.log(-np.log(1-PC))))
            CI[iConfP,1] = np.exp(thetaMax[0]+thetaMax[1]/C*(np.log(-np.log(1-pCorrMax))-np.log(-np.log(1-PC))))
        elif sigName in ['tdist','student','heavytail','neg_tdist','neg_student','neg_heavytail']:
            CI[iConfP,0] = (_my_t1icdf(pCorrMin)-_my_t1icdf(PC))*thetaMin[1] / C + thetaMin[0]
            CI[iConfP,1] = (_my_t1icdf(pCorrMax)-_my_t1icdf(PC))*thetaMax[1] / C + thetaMax[0]
        else:
             raise ValueError('unknown sigmoid function')
        
        if (pCorrMin>1) | (pCorrMin<0):
            CI[iConfP,0] = np.nan
        
        if (pCorrMax>1) | (pCorrMax<0):
            CI[iConfP,1] = np.nan
            
    return (threshold,CI)
    

def biasAna(data1, data2,options=dict()):
    """ function biasAna(data1,data2,options)
    runs a short analysis to see whether two 2AFC datasets have a bias and
    whether it can be explained with a "finger bias"-> a bias in guessing """

    options = _deepcopy(options)
    options['borders'] = np.empty([5,2])
    options['borders'][:] = np.nan
    options['expType'] = 'YesNo'

    options['priors'] = [None]*5
    options['priors'][3] = lambda x: scipy.stats.beta.pdf(x,2,2)    
    options['borders'][2,:] = np.array([0,.1])
    options['borders'][3,:] = np.array([.11,.89])
    options['fixedPars'] = np.ones([5,1])*np.nan
    options['fixedPars'][4] = 0
    options['stepN']   = np.array([40,40,40,40,1])
    options['mbStepN'] = np.array([30,30,20,20,1])

    resAll = psignifit(np.append(data1, data2, axis=0),options)
    res1 = psignifit(data1,options)
    res2 = psignifit(data2,options)

    plot.plt.figure()
    a1 = plot.plt.axes([0.15,4.35/6,0.75,1.5/6])

    plot.plotPsych(resAll,showImediate=False)
    
    plot.plotPsych(res1, lineColor= [1,0,0], dataColor = [1,0,0],showImediate=False)
    plot.plotPsych(res2,lineColor= [0,0,1], dataColor = [0,0,1],showImediate=False)
    plot.plt.ylim([0,1])

    a2 = plot.plt.axes([0.15,3.35/6,0.75,0.5/6])

    plot.plotMarginal(resAll,dim = 0,prior = False, CIpatch = False, lineColor = [0,0,0],showImediate=False)
    
    plot.plotMarginal(res1,dim = 0,lineColor = [1,0,0],showImediate=False)
    plot.plotMarginal(res2,dim = 0,lineColor=[0,0,1],showImediate=False)
    a2.relim()
    a2.autoscale_view()

    a3 = plot.plt.axes([0.15,2.35/6,0.75,0.5/6])
    plot.plotMarginal(resAll,dim = 1,prior = False, CIpatch=False, lineColor = [0,0,0],showImediate=False)

    plot.plotMarginal(res1,dim = 1,lineColor=[1,0,0],showImediate=False)
    plot.plotMarginal(res2,dim = 1,lineColor=[0,0,1],showImediate=False)
    a3.relim()
    a3.autoscale_view()

    a4 = plot.plt.axes([0.15,1.35/6,0.75,0.5/6])

    plot.plotMarginal(resAll,dim = 2, prior = False, CIpatch = False, lineColor = [0,0,0],showImediate=False)
    
    plot.plotMarginal(res1,dim = 2, lineColor=[1,0,0],showImediate=False)
    plot.plotMarginal(res2,dim=2, lineColor=[0,0,1],showImediate=False)
    a4.relim()
    a4.autoscale_view()
    
    a5 = plot.plt.axes([0.15,0.35/6,0.75,0.5/6])

    plot.plotMarginal(resAll,dim = 3, prior = False, CIpatch = False, lineColor = [0,0,0],showImediate=False)
    
    plot.plotMarginal(res1,dim = 3, lineColor=[1,0,0],showImediate=False)
    plot.plotMarginal(res2,dim = 3, lineColor=[0,0,1],showImediate=False)
    a5.set_xlim([0,1])
    a5.relim()
    a5.autoscale_view()
    
    plot.plt.draw()
    
def getDeviance(result,Nsamples=None):
    fit = result['Fit']
    data = result['data']
    pPred = fit[3] + (1-fit[2]-fit[3]) * result['options']['sigmoidHandle'](data[:,0], fit[0], fit[1])
    
    pMeasured = data[:,1]/data[:,2]
    loglikelihoodPred = data[:,1]*np.log(pPred)+(data[:,2]-data[:,1])*np.log((1-pPred))
    loglikelihoodMeasured = data[:,1]*np.log(pMeasured)+(data[:,2]-data[:,1])*np.log((1-pMeasured))
    loglikelihoodMeasured[pMeasured==1] = 0;
    loglikelihoodMeasured[pMeasured==0] = 0;

    devianceResiduals = -2*np.sign(pMeasured-pPred)*(loglikelihoodMeasured - loglikelihoodPred)
    deviance = np.sum(np.abs(devianceResiduals))
    
    if Nsamples is None:
        return devianceResiduals,deviance
    else: 
        samples_devianceResiduals = np.zeros((Nsamples,data.shape[0]))
        for iData in range(data.shape[0]):
            samp_dat = np.random.binomial(data[iData,2],pPred[iData],Nsamples)
            pMeasured = samp_dat/data[iData,2]
            loglikelihoodPred = samp_dat*np.log(pPred[iData])+(data[iData,2]-samp_dat)*np.log(1-pPred[iData])
            loglikelihoodMeasured = samp_dat*np.log(pMeasured)+(data[iData,2]-samp_dat)*np.log(1-pMeasured)
            loglikelihoodMeasured[pMeasured==1] = 0
            loglikelihoodMeasured[pMeasured==0] = 0
            samples_devianceResiduals[:,iData] = -2*np.sign(pMeasured-pPred[iData])*(loglikelihoodMeasured - loglikelihoodPred)
        samples_deviance = np.sum(np.abs(samples_devianceResiduals),axis=1)
        return devianceResiduals,deviance,samples_devianceResiduals,samples_deviance


if __name__ == "__main__":
    import sys
    psignifit(sys.argv[1], sys.argv[2])

