import numpy as np
from defines import TWO_PI
from defines import EPS


def nl(x,gamma):
    """
    Nonlinearity of the form

    .. math::

        f_{\\gamma}(x) = \\frac{1}{1-\\gamma x}

    """
    return 1/(1-gamma*x)


def f(x, gamma):
    """
    Nonlinearity of the form

    .. math::

        f_{\\gamma}(x) = \\frac{x}{1-\\sqrt{\\gamma} x} \\frac{1}{1-\\sqrt{\\gamma} \\bar{x}}

    """
    sq = np.sqrt(gamma)
    return x * nl(x, sq) * nl(np.conj(x), sq)



def nml(x, m=0.4, g=1.0):
    """
    Nonlinearity of the form

    .. math::

        f_{m,g}(x) = m\\;\\mathrm{tanh}\\left(g |x|\\right) \\frac{x}{|x|}

    """
    # return m * np.tanh(g*x)
    return m*np.tanh(g*(np.abs(x)+EPS))*x/(np.abs(x)+EPS)



def RK4(x, x_1, z_1, dt, diffeq):
    """
    Fourth-order Runge Kutta integration

    :param x:   current value of the input
    :type x: complex numpy array
    :param x_1:   last value of the input
    :type x_1: complex numpy array
    :param z_1:   last state of the system (oscillator)
    :type z_1: complex numpy array
    :param dt:   time step (fixed)
    :type dt: float
    :param diffeq: differential equation to be solved (should return dz/dt = f(x,t))
    :type diffeq: function

    :rtype: complex numpy array

    TODO:  this assumes a fixed time step between x and x_1
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
    return 0.5 * (1 + np.erf(z/np.sqrt(2)))


def pattern2odf(pattern, fs_odf):
    pass


def nextpow2(n):
    """
    TODO: document
    """
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i
