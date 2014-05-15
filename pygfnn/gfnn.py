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
        self.internal_conns = strength * make_connections(self.f, self.f, [1./3, 1./2, 1., 2., 3.], 0.5)

        # initial oscillators states
        self.z = np.zeros(self.f.shape, dtype=COMPLEX)



    def process_input(self, input, t, dt):
        """
        Run the GFNN for an external input.
        """
        def f(x, e):
            sq = np.sqrt(e)
            return x * nl(x, sq) * nl(np.conj(x), sq)

        def nml(x, m=0.4, g=.0):
            # return m * np.tanh(g*x)
            eps = np.spacing(1)
            return m*np.tanh(g*(np.abs(x)+eps))*x/(np.abs(x)+eps)


        # compute overall input (external signal + internal connections + efferent/afferent connections)
        # TODO: implement afferent/efferent connections

        # input pre-processing from NLTFT
        # x = f(n.e, x_stim) + f(n.e, nml(x_aff)) + f(n.e, nml(x_int)) + f(n.e, nml(x_eff));

        x_1 = 0 # initial last input
        dzdt = partial(zdot, f=self.f, params=self.params)
        TF = np.zeros((self.f.size, input.size), dtype=COMPLEX)
        for (i, x_stim) in enumerate(input):
            x = f(x_stim, self.params.e)            # process external signal (stimulus)
            x_int = self.z.dot(self.internal_conns) # get internal signal (via internal connections)
            x = x + f(nml(x_int), self.params.e)    # process internal signal
            self.z = RK4(x, x_1, self.z, dt, dzdt)
            x_1 = x
            TF[:,i] = self.z

        return TF

