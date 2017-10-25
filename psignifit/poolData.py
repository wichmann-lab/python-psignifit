# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 17:14:24 2016

 pools the data 
 function data = poolData(data, options)
 data must be a nx3 matrix of the form [stim.Level|#correct|#total]

 This function will pool trials together which differ at maximum poolxTol
 from the first one it finds, are separated by maximally poolMaxGap
 trials of other levels and at max poolMaxLength trials appart in general.

@author: root
"""
from numpy import tile, array, cumsum, append, empty
def poolData(data,options):
    
    dataDim1 = data.shape[0]
    dataDim2 = data.shape[1]
    counted = tile(False,[len(data),1])        # which elements we already counted
    gap = options['poolMaxGap']                # max gap between two trials of a block
    maxL = options['poolMaxLength']            # maximal blocklength
    xTol = options['poolxTol']                 # maximal difference to elements pooled in a block
    cTrialN = append([0],cumsum(data[:,2])) # cumulated number of trials with leading 0
    
    if dataDim2 == 4:
        cTrialN = data[:,3]
    
    i = 0
    pooledData = []
    
    while i < dataDim1:
        
        if not(counted[i]):
            curLevel = data[i,0]
            block = empty([0,3])
            j = i
            GapViolation = False
            curGap = 0
            
            while (j < dataDim1 and (cTrialN[j+1]-cTrialN[i] <= maxL) and not(GapViolation)) or (j==i):
                
                if abs(data[j,0] - curLevel) <= xTol and not(counted[j]):
                    counted[j] = 1
                    block = append(block, [data[j,:]], 0)
                    curGap = 0
                else:
                    curGap = curGap + data[j,2]
                
                if curGap > gap:
                    GapViolation = True
                j = j+1
            
            ntotal = sum(block[:,2])
            ncorrect = sum(block[:,1])
            level = sum(block[:,0]*block[:,2]/ntotal)
            pooledData.append(array([level,ncorrect, ntotal]))
        
        i = i + 1
        
    return array(pooledData).squeeze()

if __name__ == "__main__":
    import sys
    poolData(sys.argv[1], sys.argv[2])
