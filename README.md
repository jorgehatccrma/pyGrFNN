Gradient Frequency Neural Networks in Python
============================================

Author  : jorgeh
Date    : May, 2014


Dependencies
------------

 <!-- - PyDSTool -->
 - NumPy
 - SciPy
 - Matplotlib
 - Sphinx (to build docs)
 - sphinxcontrib-napoleon (to build docs)
 - Nose (only for automatic testing)
 - Numba?


Documentation
-------------

To generate the documentation I'm using [Sphinx](http://sphinx-doc.org/). The theme is the *ReadTheDocs* theme, that can be found [here](https://github.com/snide/sphinx_rtd_theme) (it includes instructions on how to install and use it)


Notes
-----
This repo was intended to use `virtualenv`, but for some reason, `virtualenv` and `matplotlib` decided to not like each other, so in the mean time I'm using the system-site-packages (at least for NumPy, SciPy and Matplotlib)


Testing
-------

I'm using `nose` for testing (`pip install nose`). To run all the tests, simply type:

    nosetests -v

in the root folder (where `pygfnn` and `test` folders are). The (optional) `-v` flag means verbose. To run only unit or functional test, use the `-w` option:

    nosetests -w ./ ./test/unit

