import numpy as np
from network import make_connections
from defines import COMPLEX
from utils import nl, RK4
from functools import partial
from oscillator import zdot


class GFNN(object):
    """
    GFNN. Currenlty only log-frequency spacing implemented
    """

    def __init__(self, params, fc=1.0, octaves_per_side=2.0, oscs_per_octave=64):
        """
        :param params: oscillator parameters
        :type params: Zparam
        :param fc: center frequency (in Hz.)
        :type fc: float
        :param octaves_per_side: number of octaves above (and below) fc
        :type octaves_per_side: float
        :param oscs_per_octave: number of oscillators per octave
        :type oscs_per_octave: int
        """
        #: array of oscillators' frequencies (in Hz)
        self.f = fc*np.logspace(-octaves_per_side,
                                octaves_per_side,
                                base=2.0,
                                num=2*oscs_per_octave*octaves_per_side+1)
        #: total number of oscillator in the network
        self.size = self.f.size

        #: oscillator parameters
        self.params = params

        strength = 1.0 # TODO: parametrize strength
        #: matrix of internal connections
        self.internal_conns = strength * make_connections(self.f,
                                                          self.f,
                                                          [1./3, 1./2, 1., 2., 3.],
                                                          0.5)

        # initial oscillators states
        self.z = np.zeros(self.f.shape, dtype=COMPLEX)



    def process_signal(self, input, t, dt):
        """
        Run the GFNN for an external input.

        :param input: input signal (stimulus)
        :type input: numpy complex array
        :param t: time vector (same shape as *input*)
        :type t: numpy float vector
        :param dt: input signal's sample period
        :type dt: float

        :return: Time-frequency representation of the input signal
        :rtype: numpy 2D array (rows index frequency and columns index time)
        """
        x_1 = 0 # initial last input
        dzdt = partial(zdot, f=self.f, params=self.params)
        self.TF = np.zeros((self.f.size, input.size), dtype=COMPLEX)
        for (i, x_stim) in enumerate(input):
            x_1 = self.process_time_step(dzdt, x_1, dt, x_stim)
            self.TF[:,i] = self.z

        return self.TF


    def process_time_step(self, dzdt, x_1, dt, x_stim=0):
        """
        :param dzdt: differential equation to solve using RK4. Must be of the form
                    *f(x, z)*, where *x* is the input and *z* the state; it must return dz/dt
        :type dzdt: function
        :param x_1: previous processed input (after combining all sources and nonlinearities)
        :type x_1: numpy complex array
        :param dt: time step
        :type dt: float
        :param x_stim: external signal (stimulus)
        :type x_stim: complex

        :return: the processed input, combining all sources (external, internal, afferent and
                 efferent) processed via nonlinearities
        :rtype: numpy complex array
        """
        def f(x, e):
            sq = np.sqrt(e)
            return x * nl(x, sq) * nl(np.conj(x), sq)

        def nml(x, m=0.4, g=.0):
            # return m * np.tanh(g*x)
            eps = np.spacing(1)
            return m*np.tanh(g*(np.abs(x)+eps))*x/(np.abs(x)+eps)

        # compute overall input (external signal + internal connections + eff/aff connections)
        # For reference: input pre-processing from NLTFT
        # x = f(n.e, x_stim) + f(n.e, nml(x_aff)) + f(n.e, nml(x_int)) + f(n.e, nml(x_eff));

        # TODO: implement afferent/efferent connections

        x = f(x_stim, self.params.e)            # process external signal (stimulus)
        x_int = self.z.dot(self.internal_conns) # get internal signal (via internal connections)
        x = x + f(nml(x_int), self.params.e)    # process internal signal
        self.z = RK4(x, x_1, self.z, dt, dzdt)  # integrate the diffeq

        return x    # return the computed input (will be used in the next time step as x_1)

