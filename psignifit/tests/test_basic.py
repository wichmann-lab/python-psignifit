import psignifit as ps
import numpy as np



def test_runPsignifit():
    data = np.array([[0.0010,   45.0000,   90.0000],
                 [0.0015,   50.0000,   90.0000],
                 [0.0020,   44.0000,   90.0000],
                 [0.0025,   44.0000,   90.0000],
                 [0.0030,   52.0000,   90.0000],
                 [0.0035,   53.0000,   90.0000],
                 [0.0040,   62.0000,   90.0000],
                 [0.0045,   64.0000,   90.0000],
                 [0.0050,   76.0000,   90.0000],
                 [0.0060,   79.0000,   90.0000],
                 [0.0070,   88.0000,   90.0000],
                 [0.0080,   90.0000,   90.0000],
                 [0.0100,   90.0000,   90.0000]])

    options = dict()
    options['sigmoidName'] = 'norm'   # choose a cumulative Gauss as the sigmoid
    options['expType']     = '2AFC'
    options['fixedPars']   = np.nan*np.ones((5,1))
    options['fixedPars'][2] = 0.01
    options['fixedPars'][3] = 0.5
    res=ps.psignifit(data,options)
    assert True



def test_runPlots():
    import matplotlib.pyplot as plt
    data = np.array([[0.0010,   45.0000,   90.0000],
                 [0.0015,   50.0000,   90.0000],
                 [0.0020,   44.0000,   90.0000],
                 [0.0025,   44.0000,   90.0000],
                 [0.0030,   52.0000,   90.0000],
                 [0.0035,   53.0000,   90.0000],
                 [0.0040,   62.0000,   90.0000],
                 [0.0045,   64.0000,   90.0000],
                 [0.0050,   76.0000,   90.0000],
                 [0.0060,   79.0000,   90.0000],
                 [0.0070,   88.0000,   90.0000],
                 [0.0080,   90.0000,   90.0000],
                 [0.0100,   90.0000,   90.0000]])

    options = dict()
    options['sigmoidName'] = 'norm'   # choose a cumulative Gauss as the sigmoid
    options['expType']     = '2AFC'
    options['fixedPars']   = np.nan*np.ones((5,1))
    options['fixedPars'][2] = 0.01
    options['fixedPars'][3] = 0.5
    res=ps.psignifit(data,options)

    ps.psigniplot.plotPsych(res,showImediate=False)
    plt.figure()
    ps.psigniplot.plotMarginal(res,0,showImediate=False)
    plt.figure()
    ps.psigniplot.plot2D(res,0,1,showImediate=False)
    plt.figure()
    #ps.psigniplot.plotsModelfit(res,showImediate=False)
    plt.close('all')
    assert True
