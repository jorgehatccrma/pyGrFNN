"""GFNN related code, as described in

Edward W. Large, Felix V. Almonte, and Marc J. Velasco.
A canonical model for gradient frequency neural networks.
Physica D: Nonlinear Phenomena, 239(12):905-911, 2010.

To Dos:
    - Implement linear frequency spacing


"""

import numpy as np
from network import make_connections
from defines import COMPLEX, FLOAT
from utils import f, nml
from functools import partial
from oscillator import zdot

class GFNN(object):
    """
    Gradient Frequency Neural Network

    Note:
        Currently only log-frequency spacing implemented

    Attributes:
        f: :class:`np.ndarray` -- ordered array of oscillators' natural frequencies (in Hz)
        size: ``int`` -- number of oscillators in the GFNN
        oscs_per_octave: ``int`` -- number of oscillators in a single octave
        z: :class:`np.ndarray` -- initial oscillators states
        zdot: ``function`` -- parametrized oscillator differential equation


    """

    def __init__(self,
                 zparams,
                 fc=1.0,
                 octaves_per_side=2.0,
                 oscs_per_octave=64):
        """ GFNN constructor

        Args:
            zparams (:class:`.Zparam`): oscillator parameters
            fc (float): GFNN center frequency (in Hz.)
            octaves_per_side (float): number of octaves above (and below) fc
            oscs_per_octave (float): number of oscillators per octave

        """

        # array of oscillators' frequencies (in Hz)
        self.f = np.asarray(fc*np.logspace(-octaves_per_side,
                                           octaves_per_side,
                                           base=2.0,
                                           num=2*oscs_per_octave*octaves_per_side+1),
                            dtype=FLOAT)

        # total number of oscillator in the network
        self.size = self.f.size

        # oscillator parameters
        self.zparams = zparams

        # initial oscillators states
        self.z = 1e-10*(1+1j)*np.ones(self.f.shape, dtype=COMPLEX)

        # oscillator differential equation
        self.zdot = partial(zdot, f=self.f, zparams=self.zparams)

        # number of oscillators per octave
        self.oscs_per_octave = oscs_per_octave


    def __repr__(self):
        return  "GrFNN:\n" \
                "\tfreq. range: {0}--{1}\n" \
                "\toscs/octave: {2}\n" \
                "\tnum_oscs:    {3}\n" \
                "\t{4}\n".format(min(self.f), max(self.f), self.oscs_per_octave,
                               self.size, self.zparams)


    def compute_input(self, z, external_inputs, x_stim=0):
        """Compute the input to a GFNN (:math:`x` in equation 15 in the cited paper)

        Args:
            z (:class:`numpy.array`): state of the GFNN at the instant when the
                input needs to be computed
            external_inputs (list): list of tuples of the form (*source_z*,
                *matrix*) where *source_z* is the state of the source GFNN and
                *matrix* is the connection matrix
            x_stim (:class:`numpy.array`): external stimulus

        Returns:
            :class:`numpy.array`:
                array of inputs, one element per oscillator in the GFNN

        Note:
            Here ``external_inputs`` refer to inter-layer connections

        """

        # process external signal (stimulus)
        x = f(x_stim, self.zparams.e)

        # process other inputs (internal, afferent and efferent)
        for (source_z, matrix) in external_inputs:
            # x_ext = source_z.dot(matrix)
            x_ext = matrix.dot(source_z)
            x = x + f(nml(x_ext), self.zparams.e)

        return x


