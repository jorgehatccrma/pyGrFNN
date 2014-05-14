import numpy as np

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
        self.fc = fc
        # array of oscillators' frequencies (in Hz)
        self.f = fc*np.logspace(-octaves_per_side,
                                octaves_per_side,
                                base=2.0,
                                num=2*oscs_per_octave*octaves_per_side+1)
        self.size = self.f.size

        print "GFNN size:", self.size
        print "GFNN range: %.3f (%.3f, %.3f)" % (np.ptp(self.f), np.min(self.f), np.max(self.f))

