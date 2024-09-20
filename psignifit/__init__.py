# -*- coding: utf-8 -*-
"""
Python psignifit - based on Matlab psignifit by Heiko Schütt with help from
Felix Wichmann, Jakob Macke and Stefan Harmeling.
"""
import os
import subprocess

# import here the main function
from .psignifit import psignifit
from ._pooling import pool_blocks
from ._configuration import Configuration
from ._result import Result