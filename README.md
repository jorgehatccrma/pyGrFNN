Gradient Frequency Neural Networks in Python
============================================

Author  : Jorge Herrera (jorgeh@ccrma.stanford.edu)
Date    : May, 2014


Installation
------------

Clone the repo to a local directory. Inside the director run

    python setup.py develop

This will install the package in development mode, so every time you pull changes to the repo, they should be automatically reflected in the installed version.

Alternatively, you could run 

    python setup.py install

which performs a normal installation.

In either case, if all goes well, all dependencies should be installed.


Dependencies
------------

 - NumPy
 - SciPy
 - Matplotlib
 - dispatcher
 - Nose (for testing)
 - Sphinx (to build docs)
 - sphinxcontrib-napoleon (to build docs)
 - sphinx_rtd_theme (to build docs)
 
(all dependencies are available via `pip`)


Documentation
-------------

Documentation is generated with [Sphinx](http://sphinx-doc.org/) directly from the docstrings in the code. The theme is the *ReadTheDocs* theme, that can be found [here](https://github.com/snide/sphinx_rtd_theme) (it includes instructions on how to install and use it)


Notes
-----
This repo was intended to use `virtualenv`, but for some reason, `virtualenv` and `matplotlib` decided to not like each other, so in the mean time I'm using the system-site-packages (at least for NumPy, SciPy and Matplotlib). That is on my local environment, but you are encouraged to use `virtualenv` if possible.


Testing
-------

Testing uses `nose` (`pip install nose`). To run all the tests, simply type:

    nosetests -v

in the root folder (where `pygfnn` and `test` folders are). The (optional) `-v` flag means verbose. To run only unit or functional test, use the `-w` option:

    nosetests -w ./ ./test/unit

