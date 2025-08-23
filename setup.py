#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""
from distutils.core import setup
from setuptools import setup, find_packages


packages = find_packages()
print(f"packages to be installed: {packages}")

CODENAME = "pysnid"
VERSION = '0.6.0'
        
setup(name=CODENAME,
      version=VERSION,
      description='Tools to run and read SNID',
      author='Mickael Rigault',
      author_email='m.rigault@ipnl.in2p3.fr',
      url=f'https://github.com/MickaelRigault/{CODENAME}',
      packages=packages,
#      package_data={CODENAME: ['data/*']},
#      scripts=["bin/__.py",]
     )
# End of setupy.py ========================================================


