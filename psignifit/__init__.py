# -*- coding: utf-8 -*-
"""
Python psignifit - based on Matlab psignifit by Heiko Schütt with help from
Felix Wichmann, Jakob Macke and Stefan Harmeling.
"""
import os
import subprocess
from .psignifit import *

__name__ =        'psignifit'
__description__ = 'toolbox for Bayesian psychometric function estimation'
__version__ =     '0.1'
__url__ =         'https://github.com/wichmann-lab/python-psignifit'
__author__ =      'Sophie Laturnus <sophie.laturnus@uni-tuebingen.de> www.wichmann-lab.org'
__license__ =     'GPLv3+'
__revision__ =    'N/A'

# get the HEAD sha from git if we are in a git repo (only useful for devs)
try:
    CWD = os.path.abspath(os.path.dirname(__file__))
    output = subprocess.check_output(['git', 'describe', '--tags', '--dirty=+'],
                                     cwd=CWD, stderr=subprocess.PIPE)
    __revision__ = output.strip()
except Exception:
    pass
