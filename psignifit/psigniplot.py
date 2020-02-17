# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 17:34:08 2016



@author: Ole
"""

import numpy as np
from scipy.signal import convolve as _convn
import matplotlib.pyplot as plt
import matplotlib.colors as _mcolors
from matplotlib import cm as _cm
from matplotlib.ticker import ScalarFormatter

from .marginalize import marginalize
from . import utils as _utils

def plotPsych(result,
              dataColor      = [0, 105./255, 170./255],
              plotData       = True,
              lineColor      = [0, 0, 0],
              lineWidth      = 2,
              xLabel         = 'Stimulus Level',
              yLabel         = 'Proportion Correct',
              labelSize      = 15,
              fontSize       = 10,
              fontName       = 'Helvetica',
              tufteAxis      = False,
              plotAsymptote  = True,
              plotThresh     = True,
              aspectRatio    = False,
              extrapolLength = .2,
              CIthresh       = False,
              dataSize       = 0,
              axisHandle     = None,
              showImediate   = True):
    """
    This function produces a plot of the fitted psychometric function with 
    the data.
    """
    
    fit = result['Fit']
    data = result['data']
    options = result['options']
    
    if axisHandle == None: axisHandle = plt.gca()
    try:
        plt.sca(axisHandle)
    except TypeError:
        raise ValueError('Invalid axes handle provided to plot in.')
    
    if np.isnan(fit[3]): fit[3] = fit[2]
    if data.size == 0: return
    if dataSize == 0: dataSize = 10000. / np.sum(data[:,2])
    
    if 'nAFC' in options['expType']:
        ymin = 1. / options['expN']
        ymin = min([ymin, min(data[:,1] / data[:,2])])
    else:
        ymin = 0
    
    
    # PLOT DATA
    #holdState = plt.ishold()
    #if not holdState: 
    #    plt.cla()
    #    plt.hold(True)
    xData = data[:,0]
    if plotData:
        yData = data[:,1] / data[:,2]
        markerSize = np.sqrt(dataSize/2 * data[:,2])
        for i in range(len(xData)):
            plt.plot(xData[i], yData[i], '.', ms=markerSize[i], c=dataColor, clip_on=False)
    
    # PLOT FITTED FUNCTION
    if options['logspace']:
        xMin = np.log(min(xData))
        xMax = np.log(max(xData))
        xLength = xMax - xMin
        x       = np.exp(np.linspace(xMin, xMax, num=1000))
        xLow    = np.exp(np.linspace(xMin - extrapolLength*xLength, xMin, num=100))
        xHigh   = np.exp(np.linspace(xMax, xMax + extrapolLength*xLength, num=100))
        axisHandle.set_xscale('log')
    else:
        xMin = min(xData)
        xMax = max(xData)
        xLength = xMax - xMin
        x       = np.linspace(xMin, xMax, num=1000)
        xLow    = np.linspace(xMin - extrapolLength*xLength, xMin, num=100)
        xHigh   = np.linspace(xMax, xMax + extrapolLength*xLength, num=100)
    
    fitValuesLow  = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](xLow,  fit[0], fit[1]) + fit[3]
    fitValuesHigh = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](xHigh, fit[0], fit[1]) + fit[3]
    fitValues     = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](x,     fit[0], fit[1]) + fit[3]
    
    plt.plot(x,     fitValues,           c=lineColor, lw=lineWidth, clip_on=False)
    plt.plot(xLow,  fitValuesLow,  '--', c=lineColor, lw=lineWidth, clip_on=False)
    plt.plot(xHigh, fitValuesHigh, '--', c=lineColor, lw=lineWidth, clip_on=False)
    
    # PLOT PARAMETER ILLUSTRATIONS
    # THRESHOLD
    if plotThresh:
        if options['logspace']:
            x = [np.exp(fit[0]), np.exp(fit[0])]
        else:
            x = [fit[0], fit[0]]
        y = [ymin, fit[3] + (1 - fit[2] - fit[3]) * options['threshPC']]
        plt.plot(x, y, '-', c=lineColor)
    # ASYMPTOTES
    if plotAsymptote:
        plt.plot([min(xLow), max(xHigh)], [1-fit[2], 1-fit[2]], ':', c=lineColor, clip_on=False)
        plt.plot([min(xLow), max(xHigh)], [fit[3], fit[3]],     ':', c=lineColor, clip_on=False)
    # CI-THRESHOLD
    if CIthresh:
        CIs = result['conf_Intervals']
        y = np.array([fit[3] + .5*(1 - fit[2] - fit[3]) for i in range(2)])
        plt.plot(CIs[0,:,0],               y,               c=lineColor)
        plt.plot([CIs[0,0,0], CIs[0,0,0]], y + [-.01, .01], c=lineColor)
        plt.plot([CIs[0,1,0], CIs[0,1,0]], y + [-.01, .01], c=lineColor)
    
    #AXIS SETTINGS
    plt.axis('tight')
    plt.tick_params(labelsize=fontSize)
    plt.xlabel(xLabel, fontname=fontName, fontsize=labelSize)
    plt.ylabel(yLabel, fontname=fontName, fontsize=labelSize)
    if aspectRatio: axisHandle.set_aspect(2/(1 + np.sqrt(5)))

    plt.ylim([ymin, 1])
    # tried to mimic box('off') in matlab, as box('off') in python works differently
    plt.tick_params(direction='out',right=False,top=False)
    for side in ['top','right']: axisHandle.spines[side].set_visible(False)
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.ticklabel_format(style='sci',scilimits=(-2,4))
    
    #plt.hold(holdState)
    if (showImediate):
        plt.show(0)
    return axisHandle

def plotsModelfit(result,
              showImediate   = True):
    """
    Plots some standard plots, meant to help you judge whether there are
    systematic deviations from the model. We dropped the statistical tests
    here though.
    
    The left plot shows the psychometric function with the data. 
    
    The central plot shows the Deviance residuals against the stimulus level. 
    Systematic deviations from 0 here would indicate that the measured data
    shows a different shape than the fitted one.
    
    The right plot shows the Deviance residuals against "time", e.g. against
    the order of the passed blocks. A trend in this plot would indicate
    learning/ changes in performance over time. 
    
    These are the same plots as presented in psignifit 2 for this purpose.
    """
    
    fit = result['Fit']
    data = result['data']
    options = result['options']
    
    minStim = min(data[:,0])
    maxStim = max(data[:,0])
    stimRange = [1.1*minStim - .1*maxStim, 1.1*maxStim - .1*minStim]
    
    plt.figure(figsize=(15,5))

    ax = plt.subplot(1,3,1)    
    # the psychometric function
    x = np.linspace(stimRange[0], stimRange[1], 1000)
    y = fit[3] + (1-fit[2]-fit[3]) * options['sigmoidHandle'](x, fit[0], fit[1])
    
    plt.plot(x, y, 'k', clip_on=False)
    plt.plot(data[:,0], data[:,1]/data[:,2], '.k', ms=10, clip_on=False)
    
    plt.xlim(stimRange)
    if options['expType'] == 'nAFC':
        plt.ylim([min(1./options['expN'], min(data[:,1]/data[:,2])), 1])
    else:
        plt.ylim([0,1])
    plt.xlabel('Stimulus Level',  fontsize=14)
    plt.ylabel('Percent Correct', fontsize=14)
    plt.title('Psychometric Function', fontsize=20)
    plt.tick_params(right=False,top=False)
    for side in ['top','right']: ax.spines[side].set_visible(False)
    plt.ticklabel_format(style='sci',scilimits=(-2,4))   
    
    ax = plt.subplot(1,3,2)
    # stimulus level vs deviance
    stdModel = fit[3] + (1-fit[2]-fit[3]) * options['sigmoidHandle'](data[:,0],fit[0],fit[1])
    deviance = data[:,1]/data[:,2] - stdModel
    stdModel = np.sqrt(stdModel * (1-stdModel))
    deviance = deviance / stdModel
    xValues = np.linspace(minStim, maxStim, 1000)
    
    plt.plot(data[:,0], deviance, 'k.', ms=10, clip_on=False)
    linefit = np.polyfit(data[:,0],deviance,1)
    plt.plot(xValues, np.polyval(linefit,xValues),'k-', clip_on=False)
    linefit = np.polyfit(data[:,0],deviance,2)
    plt.plot(xValues, np.polyval(linefit,xValues),'k--', clip_on=False)
    linefit = np.polyfit(data[:,0],deviance,3)
    plt.plot(xValues, np.polyval(linefit,xValues),'k:', clip_on=False)

    plt.xlabel('Stimulus Level',  fontsize=14)
    plt.ylabel('Deviance', fontsize=14)
    plt.title('Shape Check', fontsize=20)
    plt.tick_params(right=False,top=False)
    for side in ['top','right']: ax.spines[side].set_visible(False)
    plt.ticklabel_format(style='sci',scilimits=(-2,4))
    
    ax = plt.subplot(1,3,3)
    # block number vs deviance
    blockN = range(len(deviance))
    xValues = np.linspace(min(blockN), max(blockN), 1000)
    plt.plot(blockN, deviance, 'k.', ms=10, clip_on=False)
    linefit = np.polyfit(blockN,deviance,1)
    plt.plot(xValues, np.polyval(linefit,xValues),'k-', clip_on=False)
    linefit = np.polyfit(blockN,deviance,2)
    plt.plot(xValues, np.polyval(linefit,xValues),'k--', clip_on=False)
    linefit = np.polyfit(blockN,deviance,3)
    plt.plot(xValues, np.polyval(linefit,xValues),'k:', clip_on=False)
    
    plt.xlabel('Block Number',  fontsize=14)
    plt.ylabel('Deviance', fontsize=14)
    plt.title('Time Dependence?', fontsize=20)
    plt.tick_params(right=False,top=False)
    for side in ['top','right']: ax.spines[side].set_visible(False)
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.ticklabel_format(style='sci',scilimits=(-2,4))
    
    plt.tight_layout()
    if (showImediate):
        plt.show(0)


def plotMarginal(result,                  
                 dim        = 0,
                 lineColor  = [0, 105/255, 170/255],
                 lineWidth  = 2,
                 xLabel     = '',
                 yLabel     = 'Marginal Density',
                 labelSize  = 15,
                 tufteAxis  = False,
                 prior      = True,
                 priorColor = [.7, .7, .7],
                 CIpatch    = True,
                 plotPE     = True,
                 axisHandle = None,
                 showImediate   = True):
    """
    Plots the marginal for a single dimension.
    result       should be a result struct from the main psignifit routine
    dim          is the parameter to plot:
                   1=threshold, 2=width, 3=lambda, 4=gamma, 5=sigma
    """
    if isinstance(dim,str): dim = _utils.strToDim(dim)

    if len(result['marginals'][dim]) <= 1:
        print('Error: The parameter you wanted to plot was fixed in the analysis!')
        #return
    if axisHandle == None: axisHandle = plt.gca()
    try:
        plt.sca(axisHandle)
        plt.rc('text', usetex=True)
    except TypeError:
        raise ValueError('Invalid axes handle provided to plot in.')
    if not xLabel:
        if   dim == 0: xLabel = 'Threshold'
        elif dim == 1: xLabel = 'Width'
        elif dim == 2: xLabel = r'$\lambda$'
        elif dim == 3: xLabel = r'$\gamma$'
        elif dim == 4: xLabel = r'$\eta$'
    
    x        = result['marginalsX'][dim]
    marginal = result['marginals'][dim]
    CI       = np.hstack(result['conf_Intervals'][dim].T)
    Fit      = result['Fit'][dim]
    
    #holdState = plt.ishold()
    #if not holdState: plt.cla()
    #plt.hold(True)
    
    # patch for confidence region
    if CIpatch:
        xCI = np.array([CI[0], CI[1], CI[1], CI[0]])
        xCI = np.insert(xCI, 1, x[np.logical_and(x>=CI[0], x<=CI[1])])
        yCI = np.array([np.interp(CI[0], x, marginal), np.interp(CI[1], x, marginal), 0, 0])
        yCI = np.insert(yCI, 1, marginal[np.logical_and(x>=CI[0], x<=CI[1])])
        color = .5*np.array(lineColor) + .5* np.array([1,1,1])
        pat = plt.Polygon(np.array([xCI,yCI]).T, facecolor=color, edgecolor=color,alpha=1)
        axisHandle.add_patch(pat)
    
    # plot prior
    if prior:
        xprior = np.linspace(min(x), max(x), 1000)
        plt.plot(xprior, result['options']['priors'][dim](xprior), '--', c=priorColor, clip_on=False)
    
    # posterior
    plt.plot(x, marginal, lw=lineWidth, c=np.array(lineColor), clip_on=False)
    # point estimate
    if plotPE:
        plt.plot([Fit,Fit], [0, np.interp(Fit, x, marginal)], 'k', clip_on=False)
    
    
    plt.xlabel(xLabel, fontsize=labelSize, visible=True)
    plt.ylabel(yLabel, fontsize=labelSize, visible=True)
    plt.tick_params(direction='out', right=False, top=False)
    for side in ['top','right']: axisHandle.spines[side].set_visible(False)
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.ticklabel_format(style='sci', scilimits=(-2,4))
    
    #plt.hold(holdState)
    if (showImediate):
        plt.xlim([min(x), max(x)])
        plt.ylim([0, 1.1*max(marginal)])
        plt.show(0)
    return axisHandle
    

def getColorMap():
    """
       This function returns the standard University of Tuebingen Colormap. 
    """    
    midBlue = np.array([165, 30, 55])/255
    lightBlue = np.array([210, 150, 0])/255
    steps = 200
    
    MAP = _mcolors.LinearSegmentedColormap.from_list('Tuebingen', \
                    [midBlue, lightBlue, [1,1,1]],N = steps, gamma = 1.0) 
    _cm.register_cmap(name = 'Tuebingen', cmap = MAP)
    return MAP
    

    
def plotPrior(result, 
              lineWidth = 2, 
              lineColor = np.array([0,105,170])/255,
              markerSize = 30,
              showImediate   = True):
    
    """
    This function creates the plot illustrating the priors on the different 
    parameters
    """
    
    data = result['data']

    if np.size(result['options']['stimulusRange']) <= 1:
        result['options']['stimulusRange'] = np.array([min(data[:,0]), max(data[:,0])])
        stimRangeSet = False
    else:
        stimRangeSet = True
        
    stimRange = result['options']['stimulusRange']
    r = stimRange[1] - stimRange[0]
    
    # get borders for width
    # minimum = minimal difference of two stimulus levels
    
    if len(np.unique(data[:,0])) > 1 and not(stimRangeSet):
        widthmin = min(np.diff(np.sort(np.unique(data[:,0]))))
    else:
        widthmin = 100*np.spacing(stimRange[1])
    # maximum = spread of the data

    # We use the same prior as we previously used... e.g. we use the factor by
    # which they differ for the cumulative normal function
    Cfactor = (_utils.my_norminv(.95,0,1) - _utils.my_norminv(.05,0,1))/          \
            (_utils.my_norminv(1-result['options']['widthalpha'], 0,1) -   \
             _utils.my_norminv(result['options']['widthalpha'], 0,1))
    widthmax = r
    
    steps = 10000
    theta = np.empty(5)
    for itheta in range(0,5):
        if itheta == 0:
            x = np.linspace(stimRange[0]-.5*r, stimRange[1]+.5*r, steps)
        elif itheta == 1:
            x = np.linspace(min(result['X1D'][itheta]), max(result['X1D'][1],),steps)
        elif itheta == 2:
            x = np.linspace(0,.5,steps)
        elif itheta == 3:
            x = np.linspace(0,.5,steps)
        elif itheta == 4:                
            x = np.linspace(0,1,steps)
        
        y = result['options']['priors'][itheta](x)
        theta[itheta] = np.sum(x*y)/np.sum(y)
        
    if result['options']['expType'] == 'equalAsymptote':
        theta[3] = theta[2]
    if result['options']['expType'] == 'nAFC':
        theta[3] = 1/result['options']['expN']
        
    # get limits for the psychometric function plots
    xLimit = [stimRange[0] - .5*r , stimRange[1] +.5*r]
    
    """ threshold """
    
    xthresh = np.linspace(xLimit[0], xLimit[1], steps )
    ythresh = result['options']['priors'][0](xthresh)
    wthresh = _convn(np.diff(xthresh), .5*np.array([1,1])) 
    cthresh = np.cumsum(ythresh*wthresh)
    
    plt.subplot(2,3,1)
    plt.plot(xthresh,ythresh, lw = lineWidth, c= lineColor)
    #plt.hold(True)
    plt.xlim(xLimit)
    plt.title('Threshold', fontsize = 18)
    plt.ylabel('Density',  fontsize = 18)
    
    plt.subplot(2,3,4)    
    plt.plot(data[:,0], np.zeros(data[:,0].shape), 'k.', ms = markerSize*.75 )
    #plt.hold(True)
    plt.ylabel('Percent Correct', fontsize = 18)
    plt.xlim(xLimit)
    
    x = np.linspace(xLimit[0],xLimit[1],steps)
    for idot in range(0,5):
        if idot == 0:
            xcurrent = theta[0]
            color = 'k'
        elif idot == 1:
            xcurrent = min(xthresh)
            color = [1,200/255,0]
        elif idot == 2:
            tix = cthresh[cthresh >=.25].size
            xcurrent = xthresh[-tix]
            color = 'r'
        elif idot == 3:
            tix = cthresh[cthresh >= .75].size
            xcurrent = xthresh[-tix]
            color = 'b'
        elif idot == 4:
            xcurrent = max(xthresh)
            color = 'g'
        y = 100*(theta[3]+((1-theta[2])-theta[3])*result['options']['sigmoidHandle'](x,xcurrent, theta[1]))
        
        plt.subplot(2,3,4)
        plt.plot(x,y, '-', lw=lineWidth,c=color )
        plt.subplot(2,3,1)
        plt.plot(xcurrent, result['options']['priors'][0](xcurrent), '.',c=color, ms = markerSize)
    
    """ width"""
    xwidth = np.linspace(widthmin, 3/Cfactor*widthmax, steps)
    ywidth = result['options']['priors'][1](xwidth)
    wwidth = _convn(np.diff(xwidth), .5*np.array([1,1]))
    cwidth = np.cumsum(ywidth*wwidth)

    plt.subplot(2,3,2)
    plt.plot(xwidth,ywidth,lw=lineWidth,c=lineColor)
    #plt.hold(True)
    plt.xlim([widthmin,3/Cfactor*widthmax])
    plt.title('Width',fontsize=18)

    plt.subplot(2,3,5)
    plt.plot(data[:,0],np.zeros(data[:,0].size),'k.',ms =markerSize*.75)
    #plt.hold(True)
    plt.xlim(xLimit)
    plt.xlabel('Stimulus Level',fontsize=18)

    x = np.linspace(xLimit[0],xLimit[1],steps)
    for idot in range(0,5):
        if idot == 0:
            xcurrent = theta[1]
            color = 'k'
        elif idot == 1:
            xcurrent = min(xwidth)
            color = [1,200/255,0]
        elif idot == 2:
            wix = cwidth[cwidth >= .25].size
            xcurrent = xwidth[-wix]
            color = 'r'
        elif idot == 3:
            wix = cwidth[cwidth >= .75].size
            xcurrent = xwidth[-wix]
            color = 'b'
        elif idot ==4:
            xcurrent = max(xwidth)
            color = 'g'
    
        y = 100*(theta[3]+ (1-theta[2] -theta[3])* result['options']['sigmoidHandle'](x,theta[0],xcurrent))
        plt.subplot(2,3,5)
        plt.plot(x,y,'-',lw = lineWidth, c= color)
        plt.subplot(2,3,2)
        plt.plot(xcurrent,result['options']['priors'][1](xcurrent),'.',c = color,ms=markerSize)

    """ lapse """

    xlapse = np.linspace(0,.5,steps)
    ylapse = result['options']['priors'][2](xlapse)
    wlapse = _convn(np.diff(xlapse),.5*np.array([1,1]))
    clapse = np.cumsum(ylapse*wlapse)
    plt.subplot(2,3,3)
    plt.plot(xlapse,ylapse,lw=lineWidth,c=lineColor)
    #plt.hold(True)
    plt.xlim([0,.5])
    plt.title('\lambda',fontsize=18)

    plt.subplot(2,3,6)
    plt.plot(data[:,0],np.zeros(data[:,0].size),'k.',ms=markerSize*.75)
    #plt.hold(True)
    plt.xlim(xLimit)


    x = np.linspace(xLimit[0],xLimit[1],steps)
    for idot in range(0,5):
        if idot == 0:
            xcurrent = theta[2]
            color = 'k'
        elif idot == 1:
            xcurrent = 0
            color = [1,200/255,0]
        elif idot == 2:
            lix = clapse[clapse >= .25].size
            xcurrent = xlapse[-lix]
            color = 'r'
        elif idot == 3:
            lix = clapse[clapse >= .75].size
            xcurrent = xlapse[-lix]
            color = 'b'
        elif idot ==4:
            xcurrent = .5
            color = 'g'
        y = 100*(theta[3]+ (1-xcurrent-theta[3])*result['options']['sigmoidHandle'](x,theta[0],theta[1]))
        plt.subplot(2,3,6)
        plt.plot(x,y,'-',lw=lineWidth,c=color)
        plt.subplot(2,3,3)
        plt.plot(np.array(xcurrent),result['options']['priors'][2](np.array(xcurrent)),'.',c=color,ms=markerSize)
    if (showImediate):
        plt.show(0)


def plot2D(result,par1,par2, 
           colorMap = getColorMap(), 
            labelSize = 15,
            fontSize = 10,
            axisHandle = None,
            showImediate   = True):
    """ 
    This function constructs a 2 dimensional marginal plot of the posterior
    density. This is the same plot as it is displayed in plotBayes in an
    unmodifyable way.

    The result struct is passed as result.
    par1 and par2 should code the two parameters to plot:
        0 = threshold
        1 = width
        2 = lambda
        3 = gamma
        4 = eta
        
    Further plotting options may be passed.
    """
    # convert strings to dimension number
    par1,label1 = _utils.strToDim(str(par1))
    par2,label2 = _utils.strToDim(str(par2))

    assert (par1 != par2), 'par1 and par2 must be different numbers to code for the parameters to plot'
    
    if axisHandle == None:
        axisHandle = plt.gca()
        
    try:
        plt.sca(axisHandle)
    except TypeError:
        raise ValueError('Invalid axes handle provided to plot in.')

    plt.set_cmap(colorMap)
    
    marg, _, _ = marginalize(result, np.array([par1, par2]))
    
    if par1 > par2 :
        marg = marg.T


    if 1 in marg.shape:
        if len(result['X1D'][par1])==1:
            plotMarginal(result,par2)
        else:
            plotMarginal(result,par2)
    else:
        e = [result['X1D'][par2][0], result['X1D'][par2][-1], \
             result['X1D'][par1][0], result['X1D'][par1][-1]]
        plt.imshow(marg, extent = e)
        plt.ylabel(label1,fontsize = labelSize)
        plt.xlabel(label2,fontsize = labelSize)
        
    plt.tick_params(direction='out',right=False,top=False)
    for side in ['top','right']: axisHandle.spines[side].set_visible(False)
    plt.ticklabel_format(style='sci',scilimits=(-2,4))
    if (showImediate):
        plt.show(0)
