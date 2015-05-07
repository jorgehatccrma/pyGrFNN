.. pyGrFNN documentation master file, created by
   sphinx-quickstart on Tue May 13 19:30:43 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyGrFNN documentation
=====================

**pyGrFNN** (pronounced "pie-griffin") is a pure Python implementation of a
Gradient Frequency Neural Network (GrFNN), introduced by Large, Almonte and
Velasco in

   Edward W. Large, Felix V. Almonte, and Marc J. Velasco.
   *A canonical model for gradient frequency neural networks.*
   **Physica D: Nonlinear Phenomena**, 239(12):905-911, 2010.

These networks are perceptual models that can be used to compute a
time-frequency representation (TFR) of an input signal (e.g. audio or an onset
signal). Unlike other TFRs, GrFNNs can generate energy at frequencies not
present in the input signal via mode-locking.

If you need to cite this package, please use the following reference:

   Jorge Herrera, Ge Wang and Edward Large. *GrFNNs as a Framework For Nonlinear
   Audio Signal Processing.* **Proceedings of the 16th International Society for
   Music Information Retrieval Conference (ISMIR 2015)**, Malaga, Spain.


.. Contents:


LICENCE
-------

.. literalinclude:: ../LICENSE


Examples
========

1. A single layer (single GrFNN) model responding to an external stimulus

.. literalinclude:: ../examples/example1.py
   :language: python
   :linenos:


2. A double layer model responding to an external stimulus. One layer is visible
(i.e. received the stimulus directly) and the other one is hidden. They are
connected via afferent and efferent connections

.. literalinclude:: ../examples/example2.py
   :language: python
   :linenos:




Modules
=======

.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 2

   oscillator
   grfnn
   network
   vis
   resonances
   defines
   utils
   todos


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

