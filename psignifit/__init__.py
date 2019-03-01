# -*- coding: utf-8 -*-
"""
Python psignifit - based on Matlab psignifit by Heiko Schütt with help from
Felix Wichmann, Jakob Macke and Stefan Harmeling.
"""
import os
import subprocess

# import here the main function
from .psignifit import psignifit, getThreshold, getSlope, getSlopePC, biasAna, psignifitFast

__name__ =        'psignifit'
__description__ = 'toolbox for Bayesian psychometric function estimation'
__version__ =     '0.1'
__url__ =         'https://github.com/wichmann-lab/python-psignifit'
__author__ =      'psignifit contributors, www.wichmann-lab.org'
__license__ =     'GPLv3+'
__revision__ =    'N/A'

# get the git SHA if we are in a git repo (only useful for devs)

# current dir
CWD = os.path.abspath(os.path.dirname(__file__))
# try two options for getting the git revision
# - nice version with tags
# - plain SHA
for cmd in ('git describe --tags --dirty=+'), ('git rev-parse HEAD'):
    try:
        proc = subprocess.check_output(cmd.split(), cwd=CWD,
                stderr=subprocess.PIPE, universal_newlines=True)
        __revision__ = proc.strip()
    except Exception:
        # ok, don't bother
        pass

# have a way to test from python
def test():
    import pytest
    pytest.main([CWD,])
