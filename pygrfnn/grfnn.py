"""GrFNN related code, as described in

Edward W. Large, Felix V. Almonte, and Marc J. Velasco.
A canonical model for gradient frequency neural networks.
Physica D: Nonlinear Phenomena, 239(12):905-911, 2010.

To Dos:
    - Implement linear frequency spacing

"""

from __future__ import division

import numpy as np

from oscillator import zdot
from defines import COMPLEX, FLOAT

import dispatch
grfnn_update_event = dispatch.Signal(providing_args=["t", "index"])


class GrFNN(object):
    """
    Gradient Frequency Neural Network

    Note:
        Currently only log-frequency spacing implemented

    Attributes:
        f: :class:`np.ndarray` -- ordered array of oscillators' natural
            frequencies (in Hz)
        size: ``int`` -- number of oscillators in the GrFNN
        stimulus_conn_type (string) -- stimulus connection type. One of the
            following: 'linear', 'active', 'all2freq' or 'all2freq'
        z: :class:`np.ndarray` -- instantaneous oscillators states
        zdot: ``function`` -- parametrized oscillator differential equation
        w :class:`np.ndarray` -- input scaling factor (defaults to self.f)

    """

    def __init__(self,
                 zparams,
                 name='',
                 frequency_range=(0.5, 8),
                 num_oscs=128,
                 stimulus_conn_type='linear',
                 z0=None,
                 w=None,
                 save_states=True,
                 **kwargs):
        """ GrFNN constructor

        Args:
            zparams (:class:`.Zparam`): oscillator parameters
            name (string): name give to the GrFNN
            frequency_range (tuple or list): lower and upper limits of the GrFNN
                frequency range
            num_oscs (int): number of oscillators in the GrFFN
            stimulus_conn_type (string): type of stimulus connection (default
                'active')
            z0 (float or :class:`array`): initial state of the GrFNN. If not
                specified, `spontaneus_amplitudes` will be used
            w (number or :class:`array`): input scaling factor (defaults to
                `self.f` if w is `None`)
            save_states (boolean): if `True`, each computed state will be saved
                in `self.Z` (i.e. TF representation history is stored)
        """

        # array of oscillators' frequencies (in Hz)
        self.f = np.logspace(np.log10(frequency_range[0]),
                             np.log10(frequency_range[1]),
                             num_oscs+1)[:-1]


        # total number of oscillator in the network
        self.size = self.f.size

        # oscillator parameters
        self.zparams = zparams

        # stimulus connection type
        self.stimulus_conn_type = stimulus_conn_type

        # initial oscillators states
        if z0 is not None:
            self.z = z0*np.ones(self.f.shape, dtype=COMPLEX)
        else:
            r0 = 0
            f0 = self.f[0]
            a, b1, b2, e = f0*zparams.alpha, f0*zparams.beta1, f0*zparams.beta2, f0*zparams.epsilon
            r = spontaneus_amplitudes(a, b1, b2, e)
            if len(r) == 0:
                r = 0
            elif len(r) > 0:
                r = r[-1]
            r0 = (r+r0) + 0.01 * np.random.standard_normal(self.f.shape)
            phi0 = 2 * np.pi * np.random.standard_normal(self.f.shape)
            self.z = r0 * np.exp(1j * 2 * np.pi * phi0, dtype=COMPLEX)

        # oscillator differential equation
        # self.zdot = partial(zdot, f=self.f, zp=self.zparams)
        self.zdot = zdot

        # input scaling factor
        if w is None:
            self.w = self.f
        else:
            self.w = w

        # GrFNN name (e.g. sensory network)
        self.name = name

        # toggle TF representation (history of GrFNN states)
        self.save_states = save_states

    def __repr__(self):
        # return "GrFNN: {}".format(self.zparams)
        return "GrFNN: {}".format(self.name)

    def prepare_Z(self, num_frames):
        self.Z = np.zeros((self.f.size, num_frames), dtype=COMPLEX)

        def update_callback(sender, **kwargs):
            self.Z[:, kwargs['index']] = sender.z

        grfnn_update_event.connect(update_callback,
                                   sender=self,
                                   weak=False)


def passiveAllFreq(x, sqe):
    # New passive function (P_new) from GrFNN-Toolbox-1.0
    return x / ((1.0 - x * sqe) * (1.0 - np.conj(x) * sqe))


def passiveAll2Freq(x, sqe):
    # passive function (P) from GrFNN-Toolbox-1.0
    return x / (1.0 - x * sqe)


def active(z, sqe):
    return 1.0 / (1.0 - np.conj(z) * sqe)


def twoFreq(z, source_z, num, den, matrix, e, weights):
    Z1, Z2 = np.meshgrid(source_z, np.conj(z))
    Z1 **= num
    Z2 **= den-1
    M = (e ** ((num + den - 2)/2.0)) * Z1 * Z2
    return weights * np.sum(matrix * M, 1)  # sum across columns


def threeFreq(z, source_z, monomials, e):
    x = np.zeros_like(z, dtype=complex)
    Z = np.hstack((z, source_z, source_z))
    for i, zi in enumerate(z):
        ind = monomials[i].indices
        if ind.shape[0] == 0:
            continue
        exs = monomials[i].exponents
        # ec = e ** ((np.sum(np.abs(exs), axis=1)-1.0)/2.0)  # -1 instead of -2 bc. d is already d-1 from resonances.monomialsForVectors
        ec = e ** ((np.sum(exs, axis=1)-1.0)/2.0)  # -1 instead of -2 bc. d is already d-1 from resonances.monomialsForVectors
        zm = np.reshape(Z[ind.T.flatten()], ind.T.shape).T
        zm[exs<0] = np.conj(zm[exs<0])
        zm[:,0] = np.conj(zm[:,0])
        # x[i] = np.sum(ec * np.prod(zm ** np.abs(exs), axis=1))
        # x[i] = np.sum(ec * np.prod(zm ** np.abs(exs), axis=1))/ind.shape[0]
        x[i] = np.sum(1 * ec * np.prod(zm ** np.abs(exs), axis=1) / np.max(np.abs(exs)))
    return x



def compute_input(layer, z, connections, x_stim=0):
    """Compute the overall input to a GrFNN (:math:`x` in equation
    15 in the cited paper)

    Args:
        layer (:class:`grfnn`): layer which will be receiving this input
        z (:class:`numpy.array`): state of the GrFNN at the instant
            when the input needs to be computed.
        connections (list): list of tuples of the form
            (*source_z*, *connection*) where *source_z* is the
            state of the source :class:.`GrFNN` and *connection* is a
            connection object (:class:`Connection`)
        x_stim (:class:`numpy.array`): external stimulus

    Returns:
        :class:`numpy.array` -- array of inputs, one element per
        oscillator in the GrFNN

    Note:
        Here ``connections`` refer to inter-layer connections,
        as well as intra-layer connections (self connected layers)

    Note:
        `z` does not necessarily correspond to `self.z`, as this method
        might be called at in "intermediate" integration (RK4) step

    """
    #print "Stimulus"
    #print x_stim

    # process external signal (stimulus)
    if layer.stimulus_conn_type == 'linear':
        x = layer.w * x_stim
    elif layer.stimulus_conn_type == 'active':
        x = layer.w * x_stim * active(z, layer.zparams.sqe)
    elif layer.stimulus_conn_type == 'allfreq':
        x = layer.w * passiveAllFreq(x_stim, layer.zparams.sqe) * \
            active(z, layer.zparams.sqe)
    elif layer.stimulus_conn_type == 'all2freq':
        x = layer.w * passiveAll2Freq(x_stim, layer.zparams.sqe) * \
            active(z, layer.zparams.sqe)
    else:
        raise Exception("Unknown stimulus connection type '{}'".format(
            layer.stimulus_conn_type))

    #print "Processed stimulus"
    #print x

    # process other inputs (internal, afferent and efferent)
    for (source_z, conn) in connections:
        matrix, conn_type = conn.matrix, conn.conn_type
        if conn_type == '1freq':
            x = x + conn.weights * matrix.dot(source_z)
        elif conn_type == '2freq':
            x = x + twoFreq(z, source_z,
                            conn.farey_num, conn.farey_den,
                            matrix,
                            layer.zparams.e,
                            conn.weights)
        elif conn_type == '3freq':
            x = x + threeFreq(z, source_z, conn.monomials, layer.zparams.e)
        elif conn_type == 'allfreq':
            x = x + conn.weights * \
                matrix.dot(passiveAll2Freq(source_z, layer.zparams.sqe)) * \
                active(z, layer.zparams.sqe)
        elif conn_type == 'all2freq':
            x = x + conn.weights * \
                matrix.dot(passiveAllFreq(source_z, layer.zparams.sqe)) * \
                active(z, layer.zparams.sqe)
        else:
            raise Exception("Unknown connection type '{}'".format(conn_type))

    #print "Total Input"
    #print x

    return x


def spontaneus_amplitudes(alpha, beta1, beta2, epsilon):
    """
    Spontaneous amplitude of fully expanded canonical model

    Args:
        alpha (float): :math:`\\alpha` parameter of the canonical model
        beta1 (float): :math:`\\beta_1` parameter of the canonical model
        beta2 (float): :math:`\\beta_2` parameter of the canonical model
        epsilon (float): :math:`\\varepsilon` parameter of the canonical model

    Returns:

    """

    if beta2 == 0 and epsilon !=0:
        epsilon = 0

    eps = np.spacing(np.single(1))

    # Find r* numerically
    r = np.roots([float(epsilon*(beta2-beta1)),
                  0.0,
                  float(beta1-epsilon*alpha),
                  0.0,
                  float(alpha),
                  0.0])

    # only unique real values
    r = np.real(np.unique(r[np.abs(np.imag(r)) < eps]))

    r = r[r >=0]  # no negative amplitude
    if beta2 > 0:
        r = r[r < 1.0/np.sqrt(epsilon)]  # r* below the asymptote only

    def slope(r):
        return alpha + 3*beta1*r**2 + \
            (5*epsilon*beta2*r**4-3*epsilon**2*beta2*r**6) / \
            ((1-epsilon*r**2)**2)

    # Take only stable r*
    ind1 = slope(r) < 0

    ind2a = slope(r) == 0
    ind2b = slope(r-eps) < 0
    ind2c = slope(r+eps) < 0
    ind2 = np.logical_and(ind2a, np.logical_and(ind2b, ind2c))

    r = r[np.logical_or(ind1, ind2)]

    return sorted(r, reverse=True)
