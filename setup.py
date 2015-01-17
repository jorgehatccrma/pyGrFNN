#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: jorgeh
# @Date:   2014-12-10 23:59:17
# @Last Modified by:   jorgeh
# @Last Modified time: 2015-01-16 22:39:18

from setuptools import setup

setup(name='pygrfnn',
      version='0.1',
      description='Grdient Frequency Neural Networks',
      url='jorge@ccrma-stanford.edu:/user/j/jorgeh/git/pygrfnn.git',
      author='Jorge Herrera',
      author_email='jorgeh@ccrma.stanford.edu',
      license='MIT',
      packages=['pygrfnn'],
      include_package_data=True,
      install_requires=[
          'numpy',
          'dispatch',
      ],
      # setup_requires=...,
      # dependency_links=...,
      zip_safe=False
      )