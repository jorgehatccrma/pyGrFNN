"""GrFNN related code, as described in

Edward W. Large, Felix V. Almonte, and Marc J. Velasco.
A canonical model for gradient frequency neural networks.
Physica D: Nonlinear Phenomena, 239(12):905-911, 2010.

To Dos:
    - Implement linear frequency spacing

"""

import numpy as np
from network import make_connections
from defines import COMPLEX, FLOAT
from utils import ff, nml
from functools import partial
from oscillator import zdot


class GrFNN(object):
    """
    Gradient Frequency Neural Network

    Note:
        Currently only log-frequency spacing implemented

    Attributes:
        f: :class:`np.ndarray` -- ordered array of oscillators' natural
            frequencies (in Hz)
        size: ``int`` -- number of oscillators in the GrFNN
        oscs_per_octave: ``int`` -- number of oscillators in a single octave
        z: :class:`np.ndarray` -- initial oscillators states
        zdot: ``function`` -- parametrized oscillator differential equation


    """

    def __init__(self,
                 zparams,
                 fc=1.0,
                 octaves_per_side=2.0,
                 oscs_per_octave=64,
                 stimulus_conn_type='active'):
        """ GrFNN constructor

        Args:
            zparams (:class:`.Zparam`): oscillator parameters
            fc (float): GrFNN center frequency (in Hz.)
            octaves_per_side (float): number of octaves above (and below) fc
            oscs_per_octave (float): number of oscillators per octave
            stimulus_conn_type (string): type of stimulus connection (defatul
                'active')

        """

        # array of oscillators' frequencies (in Hz)
        self.f = np.asarray(fc*np.logspace(-octaves_per_side,
                                           octaves_per_side,
                                           base=2.0,
                                           num=2*oscs_per_octave *
                                           octaves_per_side+1),
                            dtype=FLOAT)

        # total number of oscillator in the network
        self.size = self.f.size

        # oscillator parameters
        self.zparams = zparams

        # stimulus connection type
        self.stimulus_conn_type = stimulus_conn_type

        # initial oscillators states
        # self.z = 1e-10*(1+0j)*np.ones(self.f.shape, dtype=COMPLEX)
        r0 = 0
        r = 0  # GrFNN Toolbox uses spontAmp.m  ...
        r0 = (r+r0) + 0.01 * np.random.standard_normal(self.f.shape)
        phi0 = 2 * np.pi * np.random.standard_normal(self.f.shape)
        self.z = r0 * np.exp(1j * 2 * np.pi * phi0, dtype=COMPLEX);

        # oscillator differential equation
        self.zdot = partial(zdot, f=self.f, zp=self.zparams)

        # number of oscillators per octave
        self.oscs_per_octave = oscs_per_octave

    def __repr__(self):
        return "GrFNN:\n" \
               "\tfreq. range: {0} -- {1}\n" \
               "\toscs/octave: {2}\n" \
               "\tnum_oscs:    {3}\n" \
               "\t{4}\n".format(min(self.f),
                                max(self.f),
                                self.oscs_per_octave,
                                self.size,
                                self.zparams)

    def compute_input(self, z, connection_inputs, x_stim=0):
        """Compute the overall input to a GrFNN (:math:`x` in equation
        15 in the cited paper)

        Args:
            z (:class:`numpy.array`): state of the GrFNN at the instant
                when the input needs to be computed
            connection_inputs (list): list of tuples of the form
                (*source_z*, *matrix*, *conn_type*) where *source_z* is the
                state of the source :class:.`GrFNN`, *matrix* is the
                connection matrix (:class:`np.ndarray`), and *conn_type* is the
                type of connection (e.g. 'allfreq')
            x_stim (:class:`numpy.array`): external stimulus

        Returns:
            :class:`numpy.array` -- array of inputs, one element per
            oscillator in the GrFNN

        Note:
            Here ``connection_inputs`` refer to inter-layer connections,
            as well as intra-layer connections (self connected layers)

        """

        def passiveAllFreq(x):
            # New passive function (P_new) from GrFNN-Toolbox-1.0
            return x / ((1.0 - x * self.zparams.sqe) * (1.0 - np.conj(x) *
                self.zparams.sqe))

        def passiveAll2Freq(x):
            # passive function (P) from GrFNN-Toolbox-1.0
            return x / (1.0 - x * self.zparams.sqe)

        def active(z):
            return 1.0 / (1.0 - np.conj(z) * self.zparams.sqe)

        # process external signal (stimulus)
        if self.stimulus_conn_type == 'allfreq':
            x = self.f * active(z) * passiveAllFreq(x_stim)
        elif self.stimulus_conn_type == 'all2freq':
            x = self.f * active(z) * passiveAll2Freq(x_stim)
        elif self.stimulus_conn_type == 'active':
            x = x_stim * self.f * active(z)
        else:
            x = x_stim * self.f

        # process other inputs (internal, afferent and efferent)
        for (source_z, matrix, conn_type) in connection_inputs:
            if conn_type == '1freq':
                x = x + matrix.dot(source_z)
            elif conn_type == '2freq':
                raise "2freq connection type not implemented. Look inside \
                    GrFNN-Toolbox-1.0/zdot.m for details."
            elif conn_type == '3freq':
                raise "3freq connection type not implemented. Look inside \
                    GrFNN-Toolbox-1.0/zdot.m for details."
            elif conn_type == 'allfreq':
                x = x + matrix.dot(passiveAll2Freq(source_z)) * active(z)
            elif conn_type == 'all2freq':
                x = x + matrix.dot(passiveAllFreq(source_z)) * active(z)

        return x