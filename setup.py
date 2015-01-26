#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: jorgeh
# @Date:   2014-12-10 23:59:17
# @Last Modified by:   jorgeh
# @Last Modified time: 2015-01-17 21:05:57

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
#          'dispatch',
          'dispatcher',
      ],
      # setup_requires=...,
      # dependency_links=...,
      zip_safe=False
      )
