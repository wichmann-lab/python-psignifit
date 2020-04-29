import psignifit as ps
import numpy as np

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
#options['fixedPars']   = np.nan*np.ones(5)
#options['fixedPars'][2] = 0.01
#options['fixedPars'][3] = 0.5

fit = ps.psignifit(data,sigmoid='norm', experiment_type='2AFC')


