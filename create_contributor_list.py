#!/usr/bin/env python3
# create contributor list from git logs.
# Note: commit authors are sorted alphabetically by first name
import subprocess
import os

# current dir
CWD = os.path.abspath(os.path.dirname(__file__))
AUTHORS = os.path.join(CWD, 'CONTRIBUTORS')

out = subprocess.check_output('git log --format=format:%an'.split(), cwd=CWD,
        universal_newlines=True)
authors = sorted(set(out.split('\n')))
with open(AUTHORS, 'wt', encoding='utf-8') as f:
    f.write('\n'.join(authors))
    f.write('\n')
