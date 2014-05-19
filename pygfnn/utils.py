import numpy as np
from defines import TWO_PI
from defines import EPS

from scipy.special import erf

def nl(x,gamma):
    """
    Nonlinearity of the form

    .. math::

        f_{\\gamma}(x) = \\frac{1}{1-\\gamma x}

    Args:
        x (:class:`numpy.array`): signal
        gamma (float): Nonlinearity parameter

    """
    return 1/(1-gamma*x)


def f(x, gamma):
    """
    Nonlinearity of the form

    .. math::

        f_{\\gamma}(x) = \\frac{x}{1-\\sqrt{\\gamma} x} \\frac{1}{1-\\sqrt{\\gamma} \\bar{x}}

    Args:
        x (:class:`numpy.array`): signal
        gamma (float): Nonlinearity parameter

    """
    sq = np.sqrt(gamma)
    return x * nl(x, sq) * nl(np.conj(x), sq)



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
    return m*np.tanh(gamma*(np.abs(x)+EPS))*x/(np.abs(x)+EPS)



def RK4(x, x_1, z_1, dt, diffeq):
    """Fourth-order Runge Kutta integration

    Args:
        x (:class:`numpy.array`): current value of the input
        x_1 (:class:`numpy.array`): last value of the input
        z_1 (:class:`numpy.array`): last state of the system (oscillator)
        dt (float): time step (fixed)
        diffeq (function): differential equation to be solved (should return dz/dt = f(x,t))

    Returns:
        (:class:`numpy.array`): updated states

    ToDo:
        this assumes a fixed time step between x and x_1
    """

    xh = 0.5*(x+x_1)   # for now, linear interpolation
    dth = 0.5*dt

    k1 = diffeq(x_1, z_1)
    k2 = diffeq(xh,  z_1 + dth*k1)
    k3 = diffeq(xh,  z_1 + dth*k2)
    k4 = diffeq(x,   z_1 + dt*k3)

    return z_1 + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)


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


def pattern2odf(pattern, fs_odf):
    pass


def nextpow2(n):
    """Similarly to Matlab's ``nextpow2``, returns the power of 2 ``>= n``
    """
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i
