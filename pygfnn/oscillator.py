"""
Basic functions implementing a nonlinear neural oscillator, as described in

Edward W. Large, Felix V. Almonte, and Marc J. Velasco.
A canonical model for gradient frequency neural networks.
Physica D: Nonlinear Phenomena, 239(12):905-911, 2010.


"""

import numpy as np

PI = np.pi


class Zparam(object):
    """
    Convenience class to encapsulate oscillator parameters.
    """

    def __init__(self, alpha=1.0, beta1=-1.0, delta1=0.0, beta2=-1.0, delta2=0.0, epsilon=0.5):
        """
        The constructor takes an optional list of named oscillator parameters:
        For example:
        ::

            params = Zparam(alpha=0.1,
                            beta1=1.0,
                            beta2=-3.0,
                            delta1=0.0,
                            delta2=0.0,
                            epsilon=0.6)


        All parameters are optional (see default values in the class description)

        :param alpha: :math:`\\alpha` (defaults to: 1.0)
        :param beta1: :math:`\\beta_1` (defaults to: -1.0)
        :param beta2: :math:`\\beta_2` (defaults to: -1.0)
        :param delta1: :math:`\\delta_1` (defaults to: 0.0)
        :param delta2: :math:`\\delta_2` (defaults to: 0.0)
        :param epsilon: :math:`\\varepsilon` (defaults to: 0.5)

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



def nl(x,gamma):
    """
    Basic nonlinearity of the form :math:`\\frac{1}{1-\\gamma x}`
    """
    return 1/(1-gamma*x)



def zdot(x, z, f, params):
    """
    Single neural oscillator, as described in equation 15 the paper referenced above

    :param x: input signal
    :type x: complex numpy array
    :param z: oscillator state
    :type z: complex numpy array
    :param f: natural frequency of the oscillator
    :type f: numpy float array
    :param params: oscillator parameters: (:math:`\\alpha, \\beta_1, \\delta_1, \\beta_2, \\delta_2` and :math:`\\varepsilon)`
    :type params: Zparam

    :rtype: complex numpy array
    """

    lin = params.a + 1j*2*PI*f
    nonlin1 = params.b1*np.abs(z)**2
    nonlin2 = params.b2*params.e*np.abs(z)**4*nl(np.abs(z)**2, params.e)
    RT = x*nl(x, np.sqrt(params.e))             # passive part of the Resonant Terms
    RT = RT * nl(np.conj(z), np.sqrt(params.e))  # time the active part

    return z * (lin + nonlin1 + nonlin2) + RT


