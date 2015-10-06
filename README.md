Gradient Frequency Neural Networks in Python
============================================

 - Author  : Jorge Herrera (jorgeh@ccrma.stanford.edu)
 - Date    : May, 2014


A python implementation of Gradient Frequency Neural Networks or GrFNNs (pronounced Griffins), introduced by [Large et. al.](http://www.ccs.fau.edu/~large/Publications/LargeAlmonteVelasco2010.pdf). Dr. Large released a Matlab toolbox with his implementation (see [here](https://github.com/MusicDynamicsLab/GrFNNToolbox)). pyGrFNN was developed in parallel but in close collaboration with Dr. Large, so while there are similarities, there are are some differences as well. 

**NOTE** This package is in development, so things may or may not change in the future.


Requirements
------------

This package has been developed and tested exclusively on Python 2.7. Other that that, you will need:

 * [`numpy`](http://www.numpy.org/)
 * [`scipy`](http://www.scipy.org/)
 * [`PyDispatcher`](https://pypi.python.org/pypi/PyDispatcher/)

Optional dependencies:

 * [`matplotlib`](http://matplotlib.org/)
 * [`nose`](https://nose.readthedocs.org/en/latest/)
 * [`sphinx`](http://sphinx-doc.org/)
 * [`sphinxcontrib-napoleon`](https://pypi.python.org/pypi/sphinxcontrib-napoleon)
 * [`sphinx_rtd_theme`](http://read-the-docs.readthedocs.org/en/latest/theme.html)



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

### Build docs

    cd <pygrfnn_dir>/docs
    make html
    open _build/html/index.html

The last line is an OS X way of opening a file on a browser from the command line. On other systems, simply open `file://<absolute_path_to_pygrfnn>/docs/_build/html/index.html`.

If something goes wrong, you can optionally run `make clean` to build from scratch.


Notes
-----
This repo was intended to use `virtualenv`, but for some reason, `virtualenv` and `matplotlib` decided to not like each other, so in the mean time I'm using the system-site-packages (at least for NumPy, SciPy and Matplotlib). That is on my local environment, but you are encouraged to use `virtualenv` if possible.


Testing
-------

Testing uses `nose` (`pip install nose`). To run all the tests, simply type:

    nosetests -v

in the root folder (where `pygfnn` and `test` folders are). The (optional) `-v` flag means verbose. To run only unit or functional test, use the `-w` option:

    nosetests -w ./ ./test/unit

