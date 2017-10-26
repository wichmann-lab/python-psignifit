#
# -*- coding: utf-8 -*-
#

import ast
import os
from setuptools import setup, find_packages

# this file is used to pick the relevant metadata for setup.py
INITFILE = os.path.join('psignifit', '__init__.py')
# the directory we are in
CWD = os.path.abspath(os.path.dirname(__file__))

def parse_keyword(key):
    """Get metadata from psignifit/__init__.py using an AST"""
    with open(os.path.join(CWD, INITFILE), encoding='utf-8') as f:
        ast_tree = ast.parse(f.read())
        for node in ast.walk(ast_tree):
            if type(node) is ast.Assign:
                try:
                    if node.targets[0].id == key:
                        return node.value.s
                except:
                    pass
    # return "not available" if we didn't find the variable
    return 'N/A'

# pick the relevant keywords from the __init__.py file
metadata_vars = ('name', 'version', 'description', 'author', 'license', 'url')
metadata = dict((var, parse_keyword('__%s__'%var)) for var in metadata_vars)

# Get the long description from the README file
with open(os.path.join(CWD, 'README.md'), encoding='utf-8') as f:
    metadata['long_description'] = f.read()

setup(
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        # Here probably BSD would be better:
        # 'License :: OSI Approved :: BSD License'
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
    ],
    packages=find_packages(),
    install_requires=['scipy', 'matplotlib', 'pytest'],
    #package_data={
    #    'sample': ['package_data.dat'],
    #},
    #entry_points={
    #    'console_scripts': [
    #        'sample=sample:main',
    #    ],
    #},
    **metadata
)

