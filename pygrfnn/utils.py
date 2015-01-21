"""Utility functions
"""

from __future__ import division
import time
import logging

import numpy as np
from scipy.special import erf

from defines import TWO_PI
from defines import EPS
from defines import FLOAT

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


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


def ff(x, gamma):
    """
    Nonlinearity of the form

    .. math::

        f_{\\gamma}(x) = \\frac{x}{1-\\sqrt{\\gamma} x}
        \\frac{1}{1-\\sqrt{\\gamma} \\bar{x}}

    Args:
        x (:class:`numpy.array`): signal
        gamma (float): Nonlinearity parameter

    """
    a = np.sqrt(gamma)
    return x * nl(x, a) * nl(np.conj(x), a)


def nml(x, m=0.4, gamma=1.0):
    """
    Nonlinearity of the form

    .. math::

        f_{m,\\gamma}(x) = m\\;\\mathrm{tanh}\\left(\\gamma |x|\\right)
        \\frac{x}{|x|}

    Args:
        x (:class:`numpy.array`): signal
        m (float): gain
        gamma (float): Nonlinearity parameter
    """
    # return m * np.tanh(g*x)
    a = np.abs(x)+EPS
    return m*np.tanh(gamma*a)*x/a


def nextpow2(n):
    """Similarly to Matlab's ``nextpow2``, returns the power of 2 ``>= n``
    """
    return 2 ** np.ceil(np.log2(n))


# execution time decorator
def timed(fun):
    """Decorator to measure execution time of a function

    Args:
        fun (function): Function to be timed

    Returns:
        (function): decorated function

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
        (dtype, int): tuple (nearest value, nearest value index)
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
        tuple: `(n, d, l, e)` for numerator, denominator, level and error

    ToDo: optimize? Look into fractions module of the standard library. It
        seems to be exactly what we need. In particular, look at
        `limit_denominator()`
    """
    frac = f
    if frac > 1:
        frac = 1/f

    ln = 0
    ld = 1
    rn = 1
    rd = 1
    l = 1

    if (abs(frac - ln/ld) <= frac*pertol):
        n = ln
        d = ld
        e = abs(frac - ln/ld)
    elif (abs(frac - rn/rd) <= frac*pertol):
        n = rn
        d = rd
        e = abs(frac - rn/rd)
    else:
        cn = ln+rn
        cd = ld+rd
        l  = l + 1
        while (abs(frac - cn/cd) > frac*pertol):
            if frac > cn/cd:
                ln=cn
                ld=cd
            else:
                rn=cn
                rd=cd
            cn = ln+rn
            cd = ld+rd
            l  = l + 1
        n = cn
        d = cd
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
        tuple: `(n, d, l, e)` for numerator np.array, denominator np.array,
            level np.array and error np.array

    Note:
        Implementation note: `frompyfunc` is just syntactic sugar, but it
        doesn't speed up vector computation. There's probably a way to optimize
        this

    """
    vFarey = np.frompyfunc(fast_farey_ratio, 2, 4)
    sel = fractions > 1
    # fractions[sel] = 1.0/fractions[sel]
    n, d, l, e = vFarey(fractions, pertol)
    # n[sel], d[sel] = d[sel], n[sel]
    # fractions[sel] = 1.0/fractions[sel]
    return n.astype(FLOAT), d.astype(FLOAT), l.astype(FLOAT), e.astype(FLOAT)


