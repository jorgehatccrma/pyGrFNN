.. pyGFNN documentation master file, created by
   sphinx-quickstart on Tue May 13 19:30:43 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyGFNN's documentation!
==================================

**pyGFNN** is a python implementation of a Gradient Frequency Neural Network (GFNN), introduced by Large, Almonte and Velasco in

Edward W. Large, Felix V. Almonte, and Marc J. Velasco.
A canonical model for gradient frequency neural networks.
Physica D: Nonlinear Phenomena, 239(12):905-911, 2010.

These networks can be used to compute a time-frequency representation (TFR) of an input signal (e.g. audio or an onset signal). Unlike other TFRs, GFNNs can generate energy at frequencies not present in the input signal via mode-locking. This property allows them to show behaviors similar to ones documented in music perception literature, making them good candidates to (partially) model human perception in a computer.


The code is loosely based on the *Nonlinear Time-Frequency Transformation Workbench* (unpublished), Matlab code obtained from Mark Velasco on early 2014. Compared to that code, this code base is incomplete, as I'm primarily interested in rhythm processing. That said, the package has been designed to be easily extended. Usually the code is commented where only rhythmic specific implementations have been coded.

.. Contents:

Modules
=======

.. toctree::
   :maxdepth: 2

   defines
   oscillator
   gfnn
   network
   utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

