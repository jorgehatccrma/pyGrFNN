"""GrFNN related code, as described in

Edward W. Large, Felix V. Almonte, and Marc J. Velasco.
A canonical model for gradient frequency neural networks.
Physica D: Nonlinear Phenomena, 239(12):905-911, 2010.

"""

from __future__ import division

import numpy as np

from oscillator import zdot, Zparam
from defines import COMPLEX, FLOAT

import logging
logger = logging.getLogger('pygrfnn.grfnn')

import dispatch
grfnn_update_event = dispatch.Signal(providing_args=["t", "index"])


class GrFNN(object):
    """
    Gradient Frequency Neural Network

    Note:
        Currently only log-frequency spacing implemented

    Attributes:
        name (`string`): Name of the GrFNN (necessary when using `modelFromJSON`)
        f (:class:`numpy.ndarray`): sorted array of natural frequencies (in Hz)
        size (`int`): number of oscillators in the GrFNN
        zparams (:class:`.Zparam`): intrinsic oscillation parameters
        stimulus_conn_type (`string`): stimulus connection type. One of the
            following: `linear`, `active`, `all2freq` or `all2freq`
        z (:class:`numpy.ndarray`): instantaneous oscillators states
        zdot (`function`): parametrized oscillator differential equation
        w (:class:`numpy.ndarray`): input scaling factor (defaults to `f`)

    """

    def __init__(self,
                 zparams=None,
                 name='',
                 frequency_range=(0.5, 8),
                 num_oscs=128,
                 stimulus_conn_type='linear',
                 z0=None,
                 w=None,
                 save_states=True,
                 **kwargs):
        """ **GrFNN constructor**

        Args:
            zparams (:class:`.Zparam` or ``dict``): oscillator intrinsic parameters
            name (``string``): name give to the GrFNN
            frequency_range (``iterable``): lower and upper limits of the GrFNN
                frequency range
            num_oscs (``int``): number of oscillators in the GrFFN
            stimulus_conn_type (``string``): type of **external stimulus** connection
                (default 'active')
            z0 (float or :class:`numpy.ndarray`): initial state of the GrFNN. If not
                specified, `spontaneus_amplitudes` will be used
            w (number or :class:`numpy.ndarray`): input scaling factor (defaults to
                `self.f` if w is `None`)
            save_states (``bool``): if `True`, each computed state will be saved
                in `self.Z` (i.e. TF representation history is stored)

        Note:
            **Stimulus Connection** only refers to how **external stimulus** is
            coupled. The overall input to a GrFNN is a combination of external
            stimulus and inter- and intra-layer coupling (see
            :func:`compute_input`). Inter- and intra-layer connection types are
            specified in :meth:`.Model.connect_layers`.

        ToDo:
            Describe initialization of ``self.z``

        """

        # if zparams is not a Zparam instance, assume it's a dictionary
	if not isinstance(zparams, Zparam):
            zparams = Zparam(**zparams)

        # array of oscillators' frequencies (in Hz)
        self.f = np.logspace(np.log10(frequency_range[0]),
                             np.log10(frequency_range[1]),
                             num_oscs)


        # total number of oscillator in the network
        self.size = self.f.size

        # oscillator parameters
        self.zparams = zparams

        # stimulus connection type
        self.stimulus_conn_type = stimulus_conn_type

        # initial oscillators states
        if z0 is not None:
            self.z = z0*np.ones(self.f.shape, dtype=COMPLEX)
        else:  # initialize using spontaneous amplitude
            r0 = 0
            f0 = self.f[0]  # FIXME: This is a hack!
            a, b1, b2, e = (f0*zparams.alpha, f0*zparams.beta1,
                            f0*zparams.beta2, f0*zparams.epsilon)
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

        logger.info('Created GrFNN with params {}'.format(self.zparams))


    def __repr__(self):
        # return "GrFNN: {}".format(self.zparams)
        return "GrFNN: {}".format(self.name)

    def _prepare_Z(self, num_frames):
        """
        Allocate a 2D array to handle the time-frequency representation

        (it is internally called upon running Model.run(), if the GrFNN's
        `save_states == True`)

        """
        self.Z = np.zeros((self.f.size, num_frames), dtype=COMPLEX)

        def update_callback(sender, **kwargs):
            self.Z[:, kwargs['index']] = sender.z

        grfnn_update_event.connect(update_callback,
                                   sender=self,
                                   weak=False)
        logger.debug('Created 2D array for TFR storage in {}'.format(self.name))


def passiveAllFreq(x, sqe):
    """
    Compute passive coupling, (multi frequency signal, of unknown frequencies):

    .. math::
        \\mathcal{P}(x,\\varepsilon) = \\frac{x}{1-x\\sqrt{\\varepsilon}}
        \\frac{1}{1-\\bar{x}\\sqrt{\\varepsilon}}
    """
    return x / ((1.0 - x * sqe) * (1.0 - np.conj(x) * sqe))


def passiveAll2Freq(x, sqe):
    """
    Compute passive coupling, (single frequency signal, of unknown frequency):

    .. math::
        \\mathcal{P}(x,\\varepsilon) = \\frac{x}{1-x\\sqrt{\\varepsilon}}
    """
    return x / (1.0 - x * sqe)


def active(z, sqe):
    """
    Compute active coupling:

    .. math::
        \\mathcal{A}(z,\\varepsilon) = \\frac{1}{1-\\bar{z}\\sqrt{\\varepsilon}}
    """
    return 1.0 / (1.0 - np.conj(z) * sqe)


def twoFreq(z, source_z, num, den, matrix, e):
    """
    Compute 2-frequency coupling

    .. math::
        n_{ij} f_j \\approx d_{ij} f_{i}

    ToDo:
        Improve documentation (add coupling equation)
    """
    Z1, Z2 = np.meshgrid(source_z, np.conj(z))
    Z1 **= num
    Z2 **= den-1
    M = (e ** ((num + den - 2)/2.0)) * Z1 * Z2
    return np.sum(matrix * M, 1)  # sum across columns


def threeFreq(z, source_z, monomials, e, conn_matrix):
    """
    Compute 3-frequency coupling

    .. math::
        n_{ij1} f_{j1} + n_{ij2} f_{j2} \\approx d_{ij1j2} f_{i}

    ToDo:
        Improve documentation (add coupling equation)
    """
    x = np.zeros_like(z, dtype=complex)
    Z = np.hstack((source_z, source_z, z))

    Lsource, Ltarget = len(source_z), len(z)
    assert (Ltarget, Lsource) == conn_matrix.shape

    for i, zi in enumerate(z):
        ind = monomials[i].indices
        if ind.shape[0] == 0:
            continue
        exs = monomials[i].exponents
        ec = e ** ((np.sum(np.abs(exs), axis=1)-1.0)/2.0)  # -1 instead of -2 bc. d is already d-1 from resonances.threeFreqMonomials
        zm = np.reshape(Z[ind.T.flatten()], ind.T.shape).T

        # get c_ij1 * c_ij2
        cij1 = conn_matrix[ind[:,2]-2*Lsource, ind[:,0]]
        cij2 = conn_matrix[ind[:,2]-2*Lsource, ind[:,0]-Lsource]
        cij = cij1 * cij2

        zm[exs<0] = np.conj(zm[exs<0])  # conjugate where there is a negative exponent
        zm[:,2] = np.conj(zm[:,2])      # also conjugate the last column (z_i)
        x[i] = np.sum(ec * cij * np.prod(zm ** np.abs(exs), axis=1))
        # x[i] = np.sum(ec * np.prod(zm ** np.abs(exs), axis=1)) / ind.shape[0]
        # x[i] = np.sum(ec * np.prod(zm ** np.abs(exs), axis=1) / np.max(np.abs(exs)))
    return x



def compute_input(layer, z, connections, x_stim=0):
    """
    Compute the overall input to a GrFNN (see
    :math:`\\mathcal{X}(x, z, \\varepsilon)` in :func:`oscillator.zdot`)

    Args:
        layer (:class:`GrFNN`): layer which will be receiving this input
        z (:class:`numpy.ndarray`): state of the GrFNN at the instant
            when the input needs to be computed.
        connections (``list``): list of tuples of the form
            ``(source_z, connection)`` where ``source_z`` is the
            state of the source :class:`GrFNN` and ``connection`` is a
            :class:`.Connection` object
        x_stim (:class:`numpy.ndarray`): external stimulus

    Return:
        :class:`numpy.ndarray`: array of inputs, one element per
        oscillator in the :class:`GrFNN`

    Note:
        ``z`` does not necessarily correspond to `self.z`, as this method
        might be called at in "intermediate" integration (RK4) step. For this
        reason this is not a :class:`GrFNN` method.

    Note:
        This is one of the bottlenecks, as it is called 4 times per GrFNN per
        sample in the input

    Note:
        Here ``connections`` refer to inter- and intra-layer connections.

    """
    #print "Stimulus"
    #print x_stim

    x = 0
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

    # process coupled GrFNNs (internal, afferent and efferent
    coupling = 0
    for (source_z, conn) in connections:
        matrix, conn_type = conn.matrix, conn.conn_type
        if conn_type == '1freq':
            coupling += conn.weights * matrix.dot(source_z)

        elif conn_type == '2freq':
            coupling += conn.weights * twoFreq(z, source_z,
                                              conn.farey_num, conn.farey_den,
                                              matrix,
                                              layer.zparams.e)

        elif conn_type == '3freq':
            coupling += conn.weights * threeFreq(z, source_z,
                                                conn.monomials,
                                                layer.zparams.e,
                                                matrix)

        elif conn_type == 'allfreq':
            coupling += conn.weights * \
                matrix.dot(passiveAllFreq(source_z, layer.zparams.sqe)) * \
                active(z, layer.zparams.sqe)

        elif conn_type == 'all2freq':
            coupling += conn.weights * \
                matrix.dot(passiveAll2Freq(source_z, layer.zparams.sqe)) * \
                active(z, layer.zparams.sqe)

        else:
            raise Exception("Unknown connection type '{}'".format(conn_type))

    #print "Total Input"
    #print x

    return x + coupling


def spontaneus_amplitudes(alpha, beta1, beta2, epsilon):
    """
    Spontaneous amplitude of fully expanded canonical model

    Args:
        alpha (float): :math:`\\alpha` parameter of the canonical model
        beta1 (float): :math:`\\beta_1` parameter of the canonical model
        beta2 (float): :math:`\\beta_2` parameter of the canonical model
        epsilon (float): :math:`\\varepsilon` parameter of the canonical model

    Returns:
        :class:`numpy.ndarray`: Spontaneous amplitudes for the oscillator

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
