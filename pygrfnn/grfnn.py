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
                 oscs_per_octave=64):
        """ GrFNN constructor

        Args:
            zparams (:class:`.Zparam`): oscillator parameters
            fc (float): GrFNN center frequency (in Hz.)
            octaves_per_side (float): number of octaves above (and below) fc
            oscs_per_octave (float): number of oscillators per octave

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

        # initial oscillators states
        # self.z = 1e-10*(1+0j)*np.ones(self.f.shape, dtype=COMPLEX)
        r0 = 0
        r = 0  # GrFNN Toolbox uses spontAmp.m  ...
        r0 = (r+r0) + 0.01 * np.random.standard_normal(self.f.shape)
        phi0 = 2 * np.pi * np.random.standard_normal(self.f.shape)
        self.z = r0 * np.exp(1j * 2 * np.pi * phi0, dtype=COMPLEX);

        # oscillator differential equation
        self.zdot = partial(zdot, f=self.f, zparams=self.zparams)

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
                (*source_z*, *matrix*) where *source_z* is the state of
                the source :class:.`GrFNN` and *matrix* is the
                connection matrix (:class:`np.ndarray`)
            x_stim (:class:`numpy.array`): external stimulus

        Returns:
            :class:`numpy.array` -- array of inputs, one element per
            oscillator in the GrFNN

        Note:
            Here ``connection_inputs`` refer to inter-layer connections,
            as well as intra-layer connections (self connected layers)

        """

        # # OPTION 1: FROM ORIGINAL NLTFT MATLAB CODE
        # # process external signal (stimulus)
        # x = ff(x_stim, self.zparams.e)
        # # process other inputs (internal, afferent and efferent)
        # for (source_z, matrix) in connection_inputs:
        #     # x_ext = source_z.dot(matrix)
        #     x_ext = matrix.dot(source_z)
        #     x = x + f(nml(x_ext), self.zparams.e)
        # return x

        # # OPTION 2: My own interpretation, where the inputs are linearly
        # # summed before applying the non-linearity
        # x = x_stim
        # # process other inputs (internal, afferent and efferent)
        # for (source_z, matrix) in connection_inputs:
        #     # x_ext = source_z.dot(matrix)
        #     x_ext = matrix.dot(source_z)
        #     x = x + x_ext
        # return ff(nml(x), self.zparams.e)

        # # OPTION 3: My own interpretation, where the external input is
        # # non-linearly transformed, before adding the connectivity
        # # inputs. THe result is also non-linearly transformed
        # x = nml(x_stim)
        # # process other inputs (internal, afferent and efferent)
        # for (source_z, matrix) in connection_inputs:
        #     # x_ext = source_z.dot(matrix)
        #     x_ext = matrix.dot(source_z)
        #     x = x + x_ext
        # return f(nml(x), self.zparams.e)

        # # OPTION 4: Apply a compressing non-linearity to the overall
        # # input, after "combining" external, internal, afferent and
        # # efferent contributions
        # # process external signal (stimulus)
        # x = ff(x_stim, self.zparams.e)
        # # process other inputs (internal, afferent and efferent)
        # for (source_z, matrix) in connection_inputs:
        #     # x_ext = source_z.dot(matrix)
        #     x_ext = matrix.dot(source_z)
        #     x = x + ff(nml(x_ext), self.zparams.e)
        # return nml(x)
        # # return nml(x, m=1. / np.sqrt(self.zparams.e))
        # # return nml(x, m=.8 / np.sqrt(self.zparams.e))


        # OPTION 5: FROM GrFNN Toolbox-1.0 MATLAB CODE
        # TODO: understand how this was derived. It doesn't coincide with the
        # 2010 paper. But it seems to work much better. Is there a reference?
        # TODO: implement connection types as describe in
        # GrFNN-Toolbox-1.0:Functions/zdot.m

        def passive(x):
            # Passive function from the 2010 paper
            # return x / (1.0 - x * self.zparams.sqe)
            # New passive function (P_new) from GrFNN-Toolbox-1.0
            return x / ((1.0 - x * self.zparams.sqe) * (1.0 - np.conj(x) * self.zparams.sqe))

        def active(z):
            return 1.0 / (1.0 - np.conj(z) * self.zparams.sqe)

        # process external signal (stimulus)
        x = x_stim * self.f
        # process other inputs (internal, afferent and efferent)
        for (source_z, matrix) in connection_inputs:
            x = x + self.f * matrix.dot(passive(source_z)) * active(z)
        return x