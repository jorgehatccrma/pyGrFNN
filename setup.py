#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: jorgeh
# @Date:   2014-12-10 23:59:17
# @Last Modified by:   Jorge Herrera
# @Last Modified time: 2015-01-27 23:23:07

from setuptools import setup

setup(name='pygrfnn',
      version='0.1',
      description='Gradient Frequency Neural Networks',
      url='jorgeh@ccrma-stanford.edu:/user/j/jorgeh/git/pygrfnn.git',
      author='Jorge Herrera',
      author_email='jorgeh@ccrma.stanford.edu',
      license='MIT',
      packages=['pygrfnn'],
      include_package_data=True,
      install_requires=[
          'numpy',
          'scipy',
#          'dispatch',
          'dispatcher',
      ],
      # setup_requires=...,
      # dependency_links=...,
      zip_safe=False
      )
