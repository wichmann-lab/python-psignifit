# -*- coding: utf-8 -*-
"""
get confidence intervals and region for parameters
function [conf_Intervals, confRegion]=getConfRegion(result)
This function returns the conf_intervals for all parameters and a
confidence region on the whole parameter space.

Useage
pass the result obtained from psignifit
additionally in confP the confidence/ the p-value for the interval is required
finally you can specify in CImethod how to compute the intervals
      'project' -> project the confidence region down each axis
      'stripes' -> find a threshold with (1-alpha) above it
  'percentiles' -> find alpha/2 and 1-alpha/2 percentiles
                   (alpha = 1-confP)

confP may also be a vector of confidence levels. The returned CIs
are a 5x2xN array then containting the confidence intervals at the different
confidence levels. (any shape of confP will be interpreted as a vector)

"""
import numpy as np
from .marginalize import marginalize
def getConfRegion(result):
    
    mode = result['options']['CImethod']
    d = len(result['X1D'])
    
    ''' get confidence intervals for each parameter --> marginalize'''
    conf_Intervals = np.zeros((d,2, len(result['options']['confP'])))
    confRegion = 0
    i = 0
    for iConfP in result['options']['confP']:
    
        if mode == 'project':
            order = np.array(result['Posterior'][:]).argsort()[::-1]
            Mass = result['Posterior']*result['weight']
            Mass = np.cumsum(Mass[order])        

            confRegion = np.reshape(np.array([True]*np.size(result['Posterior']),result['Posterior'].shape))            
            confRegion[order[Mass >= iConfP]] = False  
            for idx in range(0,d):
                confRegionM = confRegion
                for idx2 in range(0,d):
                    if idx != idx2:
                        confRegionM = np.any(confRegionM,idx2)
                start = result['X1D'][idx].flatten().nonzero()[0][0]  
                stop = result['X1D'][idx].flatten().nonzero()[0][-1]
                conf_Intervals[idx,:,i] = [start,stop]
        elif mode == 'stripes':
            for idx in range(0,d):
                (margin, x, weight1D) = marginalize(result, idx)
                order = np.array(margin).argsort()[::-1]
                
                Mass = margin*weight1D
                MassSort = np.cumsum(Mass[order])
                #find smallest possible percentage above confP
                confP1 = min(MassSort[MassSort > iConfP])
                confRegionM = np.reshape(np.array([True]*np.size(margin),np.shape(margin)))
                confRegionM[order[MassSort>confP1]] = False
                '''             
                Now we have the confidence regions
                put the borders between the nearest contained and the first
                not contained point
                
                we move in from the outer points to collect the half of the
                leftover confidence from each side
                '''
                startIndex = confRegionM.flatten().nonzero()[0][0]
                pleft = confP1 - iConfP
                if startIndex >1:
                    start = (x[startIndex]+x[startIndex-1])/2
                    start += pleft/2/margin[startIndex]
                else:
                    start = x[startIndex]
                    pleft *= 2
                
                stopIndex = confRegionM.flatten().nonzero()[0][-1]
                if stopIndex < len(x):
                    stop = (x[stopIndex]+x[stopIndex+1])/2
                    stop -= pleft/2/margin[stopIndex]
                else:
                    stop = x[stopIndex]
                    if startIndex > 1:
                        start += pleft/2/margin[startIndex]
                conf_Intervals[idx,:,i] = [start,stop]
        elif mode == 'percentiles':
            for idx in range(0,d):
                (margin, x, weight1D) = marginalize(result, idx)
                if len(x) == 1:
                    start = x[0]
                    stop = x[0]
                else:
                    Mass = margin*weight1D
                    cumMass = np.cumsum(Mass)
                    
                    confRegionM = np.logical_and(cumMass > (1-iConfP)/2,cumMass < (1-(1-iConfP)/2))
                    if any(confRegionM):
                        alpha = (1-iConfP)/2
                        startIndex = confRegionM.flatten().nonzero()[0][0]
                        if startIndex > 0:
                            start = (x[startIndex-1]+x[startIndex])/2 + (alpha-cumMass[startIndex-1])/margin[startIndex]
                        else:
                            start = x[startIndex] + alpha/margin[startIndex]
                        
                        stopIndex = confRegionM.flatten().nonzero()[0][-1]
                        if stopIndex < len(x):
                            stop = (x[stopIndex]+x[stopIndex+1])/2 + (1-alpha-cumMass[stopIndex])/margin[stopIndex+1]
                        else:
                            stop = x[stopIndex] - alpha/margin[stopIndex]
                    else:
                        cumMass_greq_iConfP = np.array(cumMass > (1-iConfP)/2)
                        index = cumMass_greq_iConfP.flatten().nonzero()[0][0]
                        MMid = cumMass[index] -Mass[index]/2
                        start = x[index] + ((1-iConfP)/2 - MMid)/margin[index]
                        stop = x[index] + ((1-(1-iConfP)/2)-MMid)/margin[index]
                conf_Intervals[idx,:,i] = np.array([start, stop])
                       
        else:
            raise ValueError('You specified an invalid mode')
                
        i += 1
    return conf_Intervals

if __name__ == "__main__":
    import sys
    getConfRegion(sys.argv[1], sys.argv[2])
