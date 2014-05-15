"""
Basic functions implementing a nonlinear neural oscillator, as described in

Edward W. Large, Felix V. Almonte, and Marc J. Velasco.
A canonical model for gradient frequency neural networks.
Physica D: Nonlinear Phenomena, 239(12):905-911, 2010.


"""

import numpy as np
from utils import nl

PI = np.pi
TWO_PI = 2*PI
abs = np.abs
sqrt = np.sqrt
conj = np.conj

class Zparam(object):
    """
    Convenience class to encapsulate oscillator parameters.
    """

    def __init__(self, alpha=0.0, beta1=-1.0, delta1=0.0, beta2=-0.25, delta2=0.0, epsilon=1.0):
        """
        The constructor takes an optional list of named oscillator parameters:
        For example:
        ::

            params = Zparam(alpha=0.5,
                            beta1=-10.0,
                            beta2=-3.0,
                            delta1=0.0,
                            delta2=0.0,
                            epsilon=0.6)


        All parameters are optional (see default values in the class description)

        :param alpha: :math:`\\alpha` (defaults to: 0.0)
        :param beta1: :math:`\\beta_1` (defaults to: -1.0)
        :param beta2: :math:`\\beta_2` (defaults to: -0.25)
        :param delta1: :math:`\\delta_1` (defaults to: 0.0)
        :param delta2: :math:`\\delta_2` (defaults to: 0.0)
        :param epsilon: :math:`\\varepsilon` (defaults to: 1.0)

        :type alpha: float
        :type beta1: float
        :type beta2: float
        :type delta1: float
        :type delta2: float
        :type epsilon: float

        **Instance attributes**:
         - **a** (*float*) - Dampening parameter :math:`\\alpha`
         - **b1** (*complex*) - :math:`b_1 = \\beta_1 + j\\delta_1`
         - **b2** (*complex*) - :math:`b_2 = \\beta_2 + j\\delta_2`
         - **e** (*float*) - Coupling strength :math:`\\varepsilon`

        """

        self.a = alpha
        self.b1 = beta1 + 1j*delta1
        self.b2 = beta2 + 1j*delta2
        self.e = epsilon



def zdot(x, z, f, params):
    """
    Dynamics of a neural oscillator, as described in equation 15 the paper referenced above.
    Can work with vectors, to simultaneously compute different oscillators

    :param x: input signal
    :type x: complex numpy array
    :param z: oscillator state
    :type z: complex numpy array
    :param f: natural frequency of the oscillator
    :type f: numpy float array
    :param params: oscillator parameters: :math:`\\alpha, \\beta_1, \\delta_1,
                                                \\beta_2, \\delta_2` and :math:`\\varepsilon`
    :type params: Zparam

    :rtype: complex numpy array
    """

    lin = params.a + 1j*TWO_PI*f
    nonlin1 = params.b1*abs(z)**2
    nonlin2 = params.b2*params.e*abs(z)**4*nl(abs(z)**2, params.e)
    RT = x*nl(x, sqrt(params.e))              # passive part of the Resonant Terms (RT)
    RT = RT * nl(conj(z), sqrt(params.e))  # times the active part of RT

    return z * (lin + nonlin1 + nonlin2) + RT



