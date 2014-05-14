import numpy as np


PI = np.pi
TWO_PI = 2*PI
exp = np.exp
sqrt = np.sqrt

def nl(x,gamma):
    """
    Basic nonlinearity of the form

    :math:`f_{\\gamma}(x) = \\frac{1}{1-\\gamma x}`
    """
    return 1/(1-gamma*x)



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

    xh = 0.5*(x+x_1)   # linear interpolation
    dth = 0.5*dt

    k1 = diffeq(x_1, z_1)
    k2 = diffeq(xh,  z_1 + dth*k1)
    k3 = diffeq(xh,  z_1 + dth*k2)
    k4 = diffeq(x,   z_1 + dt*k3)

    return z_1 + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)


def gaussPDF(x, mu, sigma):
    """
    Gaussian (normal) Probability Density Function:

    :math:`f(x, \\mu, \\sigma) = \\frac{1}{\\sqrt{2 \\pi \\sigma^2}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}`
    """
    return exp(-0.5 * ((x - mu)/sigma)**2) / (sqrt(TWO_PI) * sigma)