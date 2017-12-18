import psignifit as ps
import numpy as np
import matplotlib.pyplot as plt



#def test_runPsignifit():
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
options['fixedPars']   = np.nan*np.ones(5)
options['fixedPars'][2] = 0.01
options['fixedPars'][3] = 0.5
res=ps.psignifit(data,options)




def test_plotPsych():
    plt.figure()
    ps.psigniplot.plotPsych(res,showImediate=False)
    plt.close('all')
    assert True

def test_plotMarginal():
    plt.figure()
    ps.psigniplot.plotMarginal(res,0,showImediate=False)
    plt.close('all')
    assert True

def test_plot2D():
    plt.figure()
    ps.psigniplot.plot2D(res,0,1,showImediate=False)
    plt.close('all')
    assert True

def test_fixedPars():
    #Check that fit and borders are actually set to the fixed parametervalues
    assert np.all(res['Fit'][np.isfinite(options['fixedPars'])]== options['fixedPars'][np.isfinite(options['fixedPars'])])
    assert np.all(res['options']['borders'][np.isfinite(options['fixedPars']),0]==options['fixedPars'][np.isfinite(options['fixedPars'])])
    assert np.all(res['options']['borders'][np.isfinite(options['fixedPars']),1]==options['fixedPars'][np.isfinite(options['fixedPars'])])
