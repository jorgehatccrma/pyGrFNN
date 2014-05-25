"""Utility functions
"""

import numpy as np
from defines import TWO_PI
from defines import EPS

from scipy.special import erf

import time
import logging

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


def f(x, gamma):
    """
    Nonlinearity of the form

    .. math::

        f_{\\gamma}(x) = \\frac{x}{1-\\sqrt{\\gamma} x} \\frac{1}{1-\\sqrt{\\gamma} \\bar{x}}

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

        f_{m,\\gamma}(x) = m\\;\\mathrm{tanh}\\left(\\gamma |x|\\right) \\frac{x}{|x|}

    Args:
        x (:class:`numpy.array`): signal
        m (float): gain
        gamma (float): Nonlinearity parameter
    """
    # return m * np.tanh(g*x)
    a = np.abs(x)+EPS
    return m*np.tanh(gamma*a)*x/a



# def RK4(x, x_1, z_1, dt, diffeq):
#     """Fourth-order Runge Kutta integration

#     Args:
#         x (:class:`numpy.array`): current value of the input
#         x_1 (:class:`numpy.array`): last value of the input
#         z_1 (:class:`numpy.array`): last state of the system (oscillator)
#         dt (float): time step (fixed)
#         diffeq (function): differential equation to be solved (should return
#             dz/dt = f(x,t))

#     Returns:
#         (:class:`numpy.array`): updated states

#     ToDo:
#         this assumes a fixed time step between x and x_1
#     """

#     # # jorgeh's version
#     # xh = 0.5*(x+x_1)   # for now, linear interpolation
#     # dth = 0.5*dt

#     # k1 = diffeq(x_1, z_1)
#     # k2 = diffeq(xh,  z_1 + dth*k1)
#     # k3 = diffeq(xh,  z_1 + dth*k2)
#     # k4 = diffeq(x,   z_1 + dt*k3)

#     # return z_1 + dt*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0



def normalPDF(x, mu, sigma):
    """
    Normal (Gaussian) Probability Density Function:

    .. math::

        f(x, \\mu, \\sigma) = \\frac{1}{\\sigma\\sqrt{2 \\pi}} \
            e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}

    """
    return np.exp(-0.5 * ((x - mu)/sigma)**2) / (np.sqrt(TWO_PI) * sigma)


def normalCDF(x, mu, sigma):
    """
    Normal (Gaussian) Cumulative Density Function:

    .. math::

        F(x, \\mu, \\sigma) &= \\frac{1}{2\\pi}\\int_{-\\infty}^{\\frac{x-\\mu}{\\sigma}} e^{-t^2/2} dt \\\\
            &= \\Phi(\\frac{x-\\mu}{\\sigma}) \\\\
            &= \\frac{1}{2}\\left ( 1 + \\mathrm{erf} \\big ( \\frac{x - \\mu}{\\sigma \\sqrt{2}} \\big )  \\right )

    """
    z = (x-mu) / sigma;
    return 0.5 * (1 + erf(z/np.sqrt(2)))


def nextpow2(n):
    """Similarly to Matlab's ``nextpow2``, returns the power of 2 ``>= n``
    """
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i


# execution time decorator
def time_log(fun):
    """Decorator to measure execution time of a function

    Args:
        fun (function): Function to be timed

    Returns:
        (function): decorated function

    Example: ::

            import time
            from pygfnn.utils import time_log

            # decorate a function
            @time_log
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
        logging.info('%s took %s' % (fun.__name__, elapsed_str, ))
        return output
    return log_wrapper


def find_nearest(array, value):
    """Finds the nearest element (and its index)

    Args:
        array (:class:`numpy.array`): array to be searched
        value (dtype): value being searched

    Returns:
        (dtype, int): tuple (nearest value, nearest value index)
    """
    idx = (np.abs(array-value)).argmin()
    return (array[idx], idx)



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
