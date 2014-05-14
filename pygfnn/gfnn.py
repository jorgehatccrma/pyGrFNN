import numpy as np
from network import make_connections

class GFNN(object):
    """
    GFNN. Currenlty only log frequency-spacing implemented
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

        # TODO: parametrize strength
        strength = 0.5
        #: matrix of internal connections
        self.internal_conns = strength * make_connections(self.f, self.f, [1./3, 1./2, 1., 2., 3.], 0.5)



    def process_input(self, input, t, dt):
        # TODO: implement
        pass



