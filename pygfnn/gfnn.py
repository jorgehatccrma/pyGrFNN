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
        f (numpy.array): ordered array of oscillators' natural frequencies (in Hz)
        size (int): number of oscillators in the GFNN
        oscs_per_octave (int): number of oscillators in a single octave
        internal_conns (numpy.array): matrix of internal connections (rows index
            source and columns index destination)
        z (numpy.array): initial oscillators states
        dzdt (function): parametrized oscillator differential equation


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

        # matrix of internal connections
        self.internal_conns = None

        # initial oscillators states
        self.z = 1e-10*(1+1j)*np.ones(self.f.shape, dtype=COMPLEX)

        # oscillator differential equation
        self.dzdt = partial(zdot, f=self.f, zparams=self.zparams)

        # number of oscillators per octave
        self.oscs_per_octave = oscs_per_octave


    def connect_internally(self, relations=None, internal_strength=0.5,
                           internal_stdev=0.5, complex_kernel=False):
        """ Creates internal connections

        Args:
            relations (list): list of connection relations to be established.
                For example, ``relations = [0.5, 1., 3.]`` will establish the
                following connections: :math:`f_{dest} == 0.5f_{src};\\quad
                f_{dest} == f_{src};\\quad f_{dest} == 3f_{src}`. If *None*, it
                will be set to ``[1]``
            internal_strength (float): weight of the internal connection.
                If 0.0, not connections will be created
            internal_stdev (float): internal connections standard deviation.
                If *internal_strength==0.0*, this will be ignored.
            complex_kernel (bool): if *True*, the connections are complex numbers

        Warning:
            No sanity check has been implemented

        """
        if relations is None: relations = [1]

        if internal_strength > 0:
            self.internal_conns = internal_strength * \
                                  make_connections(self,
                                                   self,
                                                   relations,
                                                   internal_stdev,
                                                   complex_kernel=complex_kernel,
                                                   self_connect=False)



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
            Here *external_inputs* refer to inter-layer connections

        """

        # process external signal (stimulus)
        x = f(x_stim, self.zparams.e)

        # process internal connections
        if self.internal_conns is not None:
            # process internal signal (via internal connections)
            # x_int = z.dot(self.internal_conns)
            x_int = self.internal_conns.dot(z)
            x = x + f(nml(x_int), self.zparams.e)

        # process other external inputs (afferent / efferent)
        for (source_z, matrix) in external_inputs:
            # x_ext = source_z.dot(matrix)
            x_ext = matrix.dot(source_z)
            x = x + f(nml(x_ext), self.zparams.e)

        return x


