"""Utility functions
"""

from __future__ import division
import time
import cPickle
from functools import update_wrapper

import logging
logger = logging.getLogger('pygrfnn.utils')

import numpy as np
from scipy.special import erf

from defines import TWO_PI
from defines import EPS
from defines import FLOAT


def nl(x, gamma):
    """
    Nonlinearity of the form

    .. math::

        f_{\\gamma}(x) = \\frac{1}{1-\\gamma x}

    Args:
        x (:class:`numpy.array`): signal
        gamma (float): Nonlinearity parameter

    Note:
        The integral of ``gamma * nl(x, gamma)`` is

        .. math::

            \\int \\frac{\\gamma}{1 - \\gamma x} = -\\log (1 - \\gamma x)

    """
    return 1.0/(1.0-gamma*x)


def nextpow2(n):
    """
    Return the power of 2 ``>= n``, similarly to Matlab's ``nextpow2``
    """
    return 2 ** np.ceil(np.log2(n))


# execution time decorator
def timed(fun):
    """Decorator to measure execution time of a function

    Args:
        fun (``function``): Function to be timed

    Returns:
        ``function``: decorated function

    Example: ::

            import time
            from pygrfnn.utils import timed

            # decorate a function
            @timed
            def my_func(N, st=0.01):
                for i in range(N):
                    time.sleep(st)


            # use it as you would normally would
            my_func(100)


    """
    def log_wrapper(*args, **kwargs):
        t0 = time.time()
        output = fun(*args, **kwargs)
        elapsed = time.time() - t0
        if elapsed < 60:
            elapsed_str = '%.2f seconds' % (elapsed)
        else:
            elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        logging.info('\n%s took %s' % (fun.__name__, elapsed_str, ))
        return output
    return log_wrapper


def find_nearest(arr, value):
    """Finds the nearest element (and its index)

    Args:
        arr (:class:`numpy.array`): array to be searched
        value (dtype of list): value(s) being searched

    Returns:
        ``(dtype, int)``: tuple (nearest value, nearest value index)
    """
    # arr must be sorted
    idx = arr.searchsorted(value)
    idx = np.clip(idx, 1, len(arr)-1)
    left = arr[idx-1]
    right = arr[idx]
    idx -= value - left < right - value
    return (arr[idx], idx)


def nice_log_values(array):
    """Returns an array of logarithmically spaced values covering the range in
    *array*

    The values in the array will be only powers of 2.

    Args:
        array (:class:`numpy.array`): source array

    Returns:
        :class:`numpy.array`: log spaced nice values
    """
    low = np.log2(nextpow2(np.min(array)))
    high = np.log2(nextpow2(np.max(array)))
    nice = 2**np.arange(low, 1+high)
    return nice[(nice >= np.min(array)) & (nice <= np.max(array))]


def memoize(f):
    """ Memoization decorator for functions taking one or more arguments. """
    class memodict(dict):
        def __init__(self, f):
            self.f = f
        def __call__(self, *args):
            return self[args]
        def __missing__(self, key):
            ret = self[key] = self.f(*key)
            return ret
    return memodict(f)


class MemoizeMutable:
    """
    Decorator to memoize function with mutable arguments.
    """
    def __init__(self, fn):
        """
        Args:
            fn (``function``): function to decorate
        """
        self.fn = fn
        self.memo = {}
        update_wrapper(self, fn)  ## so sphinx can find the docstring

    def __call__(self, *args, **kwds):
        import cPickle
        str = cPickle.dumps(args, 1)+cPickle.dumps(kwds, 1)
        if not self.memo.has_key(str):
            self.memo[str] = self.fn(*args, **kwds)
        return self.memo[str]


@memoize
def fast_farey_ratio(f, pertol=0.01):
    """
    Compute the Farey ratio of a fraction f with tolerance t.

    To allow usage of single argument memoization (see `memodict`), fraction
    and tolerance are passed as a tuple

    Args:
        f (float): fraction
        pertol (float): tolerance

    Returns:
        ``(n, d, l, e)``: tuple of (numerator, denominator, level, error)

    ToDo: optimize? Look into fractions module of the standard library. It
        seems to be exactly what we need. In particular, look at
        `limit_denominator()`
    """
    frac = f
    if frac > 1:
        frac = 1/f

    ln, ld = 0, 1
    rn, rd = 1, 1
    l = 1

    if (abs(frac - ln/ld) <= frac*pertol):
        n, d = ln, ld
        e = abs(frac - ln/ld)
    elif (abs(frac - rn/rd) <= frac*pertol):
        n, d = rn, rd
        e = abs(frac - rn/rd)
    else:
        cn, cd = ln+rn, ld+rd
        l  = l + 1
        while (abs(frac - cn/cd) > frac*pertol):
            if frac > cn/cd:
                ln, ld = cn, cd
            else:
                rn, rd = cn, cd
            cn, cd = ln+rn, ld+rd
            l  = l + 1
        n, d = cn, cd
        e = abs(frac - cn/cd)

    if f > 1:
        n, d = d, n

    return n, d, l, e

def fareyratio(fractions, pertol=.01):
    """
    Compute Farey ratio for a list of fractions

    Args:
        fractions (:class:`numpy.array`): array of fractions to be simplified
        pertol (float): tolerance

    Returns:
        ``(n, d, l, e)``: tuple of 4 :class:`numpy.ndarray` (numerator,
            denominator, level, error)
    """
    # Implementation note: :class:`numpy.frompyfunc` is just syntactic sugar,
    # but it does not speed up vector computation. There's probably a way to
    # optimize this.
    vFarey = np.frompyfunc(fast_farey_ratio, 2, 4)
    sel = fractions > 1
    # fractions[sel] = 1.0/fractions[sel]
    n, d, l, e = vFarey(fractions, pertol)
    # n[sel], d[sel] = d[sel], n[sel]
    # fractions[sel] = 1.0/fractions[sel]
    return n.astype(FLOAT), d.astype(FLOAT), l.astype(FLOAT), e.astype(FLOAT)


def cartesian(arrays):
    """
    Generate a cartesian product of input arrays.

    This Implementation is faster the numpy's built-in version

    Obtained from http://goo.gl/arVNNv


    Args:
        arrays (``list`` or :class:`numpy.ndarray`) : list of array-like
            1-D arrays to form the cartesian product of.

    Returns:
        out:
            :class:`numpy.ndarray`: 2-D array of shape ``(M, len(arrays))``
            containing cartesian products formed of input arrays.
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out
