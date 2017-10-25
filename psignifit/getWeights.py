# -*- coding: utf-8 -*-
"""
creates the weights for quadrature / numerical integration
function weight=getWeights(X1D)
this function calculates the weights for integration/the volume of the
cuboids given by the 1 dimensional borders in X1D

"""
from numpy import tile, reshape, ones, array, newaxis, where, multiply, convolve
from scipy import diff
from scipy.signal import convolve as convn

def getWeights(X1D):
    
    d = len(X1D)
    
    ''' puts the X values in their respective dimensions to use bsxfun for
    evaluation'''
    Xreshape = []
    Xreshape.append(X1D[0].reshape(-1,1))
    if d >= 2:
        Xreshape.append(X1D[1].reshape([1,-1]))
    
    for idx in range (2,d):
        dims = [1]*idx
        dims.append(-1)
        Xreshape.append(reshape(X1D[idx], dims))
    
    
    # to find and handle singleton dimensions
    sizes = [X1D[ix].size for ix in range(0,d)]
    Xlength = array(sizes)
    
    ''' calculate weights '''
    #1) calculate mass/volume for each cuboid
    weight = 1
    for idx in range(0,d):
        if Xlength[idx] > 1:
            if idx > 1:
                weight = multiply(weight[...,newaxis],diff(Xreshape[idx], axis=idx))
            else:
                weight = multiply(weight,diff(Xreshape[idx], axis=idx))
        else:
            weight = weight[...,newaxis]
    #2) sum to get weight for each point
    if d > 1:
        dims = tile(2,[1,d])
        dims[0,where(Xlength == 1)] = 1
        d = sum(Xlength > 1)
        weight = (2.**(-d))*convn(weight, ones(dims.ravel()), mode='full')
    else:
        weight = (2.**(-1))*convolve(weight, array([[1],[1]]))

    return weight

if __name__ == "__main__":
    import sys
    getWeights(sys.argv[1])
