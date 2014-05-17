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

    def __init__(self, params, fc=1.0,
                 octaves_per_side=2.0,
                 oscs_per_octave=64,
                 internal_strength=0.5,
                 internal_stdev=0.5):
        """
        :param params: oscillator parameters
        :type params: Zparam
        :param fc: center frequency (in Hz.)
        :type fc: float
        :param octaves_per_side: number of octaves above (and below) fc
        :type octaves_per_side: float
        :param oscs_per_octave: number of oscillators per octave
        :type oscs_per_octave: int
        :param internal_strength: weight of the internal connection.
            If 0.0, not connections will be created
        :type internal_strength: float
        :param internal_stdev: internal connections standard deviation.
            If *internal_strength==0.0*, this will be ignored.
        :type internal_stdev: float
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

        #: matrix of internal connections
        self.internal_conns = None

        if internal_strength > 0:
            self.internal_conns = internal_strength * \
                                  make_connections(self.f,
                                                   self.f,
                                                   [1./3, 1./2, 1., 2., 3.],
                                                   internal_stdev,
                                                   self_connect=False) / oscs_per_octave    # TODO: why /oscs_per_octave?

        #: initial oscillators states
        self.z = 1e-10*(1+1j)*np.ones(self.f.shape, dtype=COMPLEX)

        #: last processed input
        self.x_1 = 0

        #: oscillator differential equation
        self.dzdt = partial(zdot, f=self.f, params=self.params)


    def reset(self):
        self.x_1 = 0;
        self.z = 1e-10*(1+1j)*np.ones(self.f.shape, dtype=COMPLEX)


    def process_signal(self, input, t, dt):
        """
        Run the GFNN for an external input. It runs isolated, not as part of a network
        (doesn't consider other inputs such as efferent or afferent).

        :param input: input signal (stimulus)
        :type input: numpy complex array
        :param t: time vector (same shape as *input*)
        :type t: numpy float vector
        :param dt: input signal's sample period
        :type dt: float

        :return: Time-frequency representation of the input signal
        :rtype: numpy 2D array (rows index frequency and columns index time)
        """

        def f(x, e):
            sq = np.sqrt(e)
            return x * nl(x, sq) * nl(np.conj(x), sq)

        def nml(x, m=0.4, g=1.0):
            # return m * np.tanh(g*x)
            eps = np.spacing(1)
            return m*np.tanh(g*(np.abs(x)+eps))*x/(np.abs(x)+eps)


        self.TF = np.zeros((self.f.size, input.size), dtype=COMPLEX)
        for (i, x_stim) in enumerate(input):
            # process external signal (stimulus)
            x = f(x_stim, self.params.e)
            # print "-"*20
            # print x

            if self.internal_conns is not None:
                # process internal signal (via internal connections)
                x_int = self.z.dot(self.internal_conns)
                x = x + f(nml(x_int), self.params.e)
            # print x


            self.process_time_step(dt, x)
            self.TF[:,i] = self.z

        return self.TF


    def process_time_step(self, dt, x):
        """
        :param dt: time step
        :type dt: float
        :param x: input
        :type x: numpy complex array
        """
        self.z = RK4(x, self.x_1, self.z, dt, self.dzdt)    # integrate the diffeq
        self.x_1 = x    # store the computed input (will be used in the next time step as x_1)

