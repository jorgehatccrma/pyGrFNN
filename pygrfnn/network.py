"""Network of GrFNNs

This module provides the necessary code to build a model connecting
multiple GrFNNs. Connections can be made within a GrFNN or between pairs
of them.

More importantly, it provides a method to run the model (process an
stimulus), including the ability to learn the connections between
GrFNNs.

To Dos:
    - Implement other types of connectivities
    - Implement learning?

"""

import sys

import numpy as np
from scipy.stats import norm
import dispatch

from utils import nl
from utils import fareyratio
from defines import COMPLEX, PI, PI_2
from grfnn import compute_input

model_update_event = dispatch.Signal(providing_args=["z", "t"])

def make_connections(source, dest, strength=1.0, range=1.02,
                     modes=None, mode_amps=None,
                     complex_kernel=False, self_connect=True):
    """Creates a connection matrix, that connects source layer to destination
    layer.

    Args:
        source (:class:`.GrFNN`): source GrFNN (connections will be made
            between this and *dest*)
        dest (:class:`.GrFNN`): destination GrFNN (connections will be
            made between *source* and *this*)
        strength (float): connection strength (multiplicative real
            factor)
        range (float): defines the standard deviation to use in the connections
            ("spread" them with neighbors). It is expressed as a ratio, to
            account for the log scale of the oscillators' frequency
        modes (:class:`numpy.array`): frequency modes to connect
            (e.g. [1/3, 1/2, 1, 2, 3]). If *None*, it will be set to
            ``[1]``
        mode_amplitudes (:class:`numpy.array`): amplitude for each mode in
            `modes` (e.g. [.5, .75, 1, .75, .5]). If *None*, it will be set to
            ``[1] * len(modes)``
        complex_kernel (bool): If *True*, the connections will be
            complex (i.e. include phase information). Otherwise, the
            connections will be real-valued weights.
        self_connect (bool): if *False*, the connection from source_f[i]
            to dest_f[j] (where source_f[i] == dest_f[j]) will be set to
            0

    ToDo:
        - Revise the units of ``stdev``
        - Possibly handle different stdevs for different modes?

    Returns:
        :class:`numpy.array`: Connection matrix (rows index destination and
            columns index source). In other words, to obtain the state at the
            destination, you must use `M.dot(source.z)`, where `M` is the
            connection matrix.

    """

    # matrix (2D array) of relative frequencies
    # source is indexed in columns and destination in rows. That is,
    # RF(i,j) specifies the relative frequency of source_f[j] w.r.t.
    # dest_f[i]
    [FS, FT] = np.meshgrid(source.f, dest.f)
    RF = FT/FS
    logRF = np.log2(RF)

    assert RF.shape == (len(dest.f), len(source.f))

    # matrix of connections
    # connection matrices index source in rows and destination in
    # columns. That is, conn(i,j) specifies the connection weight from
    # the i-th element to the j-th element
    conns = np.zeros(RF.shape, dtype=COMPLEX)

    if modes is None:
        modes = [1]

    if mode_amps is None:
        mode_amps = [1.0] * len(modes)

    assert len(modes) == len(mode_amps)

    sigmas = np.abs(np.log2(range))*np.ones(len(modes))
    log_modes = np.log2(modes)

    df = np.log2(dest.f[-1]/dest.f[0])/len(dest.f)

    # Make self connections using a Gaussian distribution
    for m, a, s in zip(log_modes, mode_amps, sigmas):
        R = a * norm.pdf(logRF, m, s) * df
        if complex_kernel:
            Q = PI_2*(2.0*norm.cdf(logRF, m, s)-1)
        else:
            Q = np.zeros(R.shape)

        if not self_connect:
            R[RF == 1] = 0
            Q[RF == 1] = 0

        conns += R * np.exp(1j*Q)

    return strength * conns


class DuplicatedLayer(Exception):
    """
    Raised when attempting to add a previously added layer to a network

    Attributes:
        layer: :class:`.GrFNN` -- duplicated layer
    """

    def __init__(self, layer):
        self.layer = layer


class UnknownLayer(Exception):
    """
    Raised when attempting to use a layer unknown to the network

    Attributes:
        layer: :class:`.GrFNN` -- unknown layer
    """

    def __init__(self, layer):
        self.layer = layer

    def __str__(self):
        return "Unknown layer %s. Did you forget to call " \
               "'add_layer(layer)'?" % (repr(self.layer))


class Cparam(object):
    """Convenience class to encapsulate connectivity learning parameters.

    lambda = .001; mu1 = -1; mu2 = -50; ceps = 16, kappa =

    Attributes:
        l: :class:`float` -- linear forgetting rate :math:`\\lambda`
        m1: :class:`float` --  non-linear forgetting rate 1 :math:`\\mu_1
        m2: :class:`float` -- non-linear forgetting rate 2 :math:`\\mu_2
        k: :class:`float` -- learning rate :math:`\\kappa`
        e: :class:`float` -- Coupling strength :math:`\\varepsilon`

    """

    def __init__(self, lmbda=-1.0, mu1=0.0, mu2=0.0,
                 kappa=1.0, epsilon=1.0):
        """Constructor.

        Args:
            lmbda (float): :math:`\\lambda` (defaults to: -1.0) (**this is not
                a typo: `lambda` is a keyword in python, so we used a slight
                variation of the word**)
            mu1 (float): :math:`\\mu_1` (defaults to: -1.0)
            mu2 (float): :math:`\\mu_2` (defaults to: -0.25)
            kappa (float): :math:`\\kappa` (defaults to: 0.0)
            epsilon (float): :math:`\\varepsilon` (defaults to: 1.0)

        """

        self.l = lmbda
        self.m1 = mu1
        self.m2 = mu2
        self.k = kappa
        self.e = epsilon
        self.sqe = np.sqrt(self.e)


    def __repr__(self):
        return  "Cparams:\n" \
                "\tlambda:   {0}\n" \
                "\tmu_1:  {1}\n" \
                "\tmu_2:  {2}\n" \
                "\tkappa:  {3}\n" \
                "\tepsilon: {4}\n".format(self.l,
                                          self.m1,
                                          self.m2,
                                          self.k,
                                          self.e)


class Connection(object):
    """
    Connection object

    Args:
        source (:class:`.GrFNN`): source layer
        destination (:class:`.GrFNN`): destination layer
        matrix (:class:`np.ndarray`): connection matrix
        conn_type (string): type of GrFNN connections to use. Possible values:
            'allfreq', 'all2freq', '1freq', '2freq', '3freq'
        weight (float): frequency weight factor
        learn_params (:class:`.Cparam`): learning params. No learning is performed
            when set to `None`
        self_connect (bool): if ``False``, the diagonal of the
            matrix is kept to 0 (even when learning is enabled)

    Attributes:
        source: :class:`.GrFNN` -- source layer
        destination: :class:`.GrFNN` -- destination layer
        matrix: :class:`np.ndarray` -- connection matrix
        cparams: :class:`.Cparam` -- Learning params (`None` means no learning)
        d: ``float`` -- "passive" learning rate (i.e. forgetting factor)
        k: ``float`` -- "active" learning rate
        RF: :class:`np.ndarray` -- array of frequency ratio.
            `R(i,j) = dest(i)/source.f(j)`
        farey_num: :class:`np.ndarray` -- Farey numerator for the
            frequency relationship RF(i,j)
        farey_den: :class:`np.ndarray` -- Farey denominator for the
            frequency relationship RF(i,j)
    """

    def __init__(self,
                 src,
                 dest,
                 matrix,
                 conn_type,
                 weight=1.0,
                 learn_params=None,
                 self_connect=True):
        self.source = src
        self.destination = dest
        self.matrix = matrix.copy()
        self.cparams = learn_params
        self.self_connect = self_connect
        self.conn_type = conn_type

        # this is only for 'log' spaced GrFNNs
        self.weights = weight * dest.f

        # compute integer relationships between frequencies of both layers
        # using Farey sequences (http://en.wikipedia.org/wiki/Farey_sequence)
        [FS, FT] = np.meshgrid(self.source.f, self.destination.f)
        self.RF = FT/FS
        self.farey_num, self.farey_den, _, _ = fareyratio(self.RF, 0.05)

        if not self.self_connect:
            self.matrix[np.logical_and(self.farey_num==1, self.farey_den==1)] = 0

    def __repr__(self):
        return "Connection from {0} " \
               "(self_connect={1})\n".format(self.source.name,
                                             self.self_connect)


class Model(object):
    """
    A network of GrFNNs.

    Different GrFNNs are referred to as layers. Layers can be added as
    visible or hidden; the former means that it will directly receive
    external stimulus, while the later implies that the inputs will
    consist only of internal connections (internal to the layer or from
    other layers in the network).

    Attributes:
        layers: ``[layer, input_channel]`` -- list of :class:`.GrFNN`
            layers and its external input channel
        connections: ``{layer: [connections]}`` -- dictionary of
            connections. *Keys* correspond to destination layers
            (:class:`.GrFNN`). *Values* are a list of connections
            (:class:`.Connection`).

    """

    def __init__(self):
        """Model constructor"""

        # list of GrFNN layers (and its corresponding external input channel)
        self._layers = []

        # connections
        self.connections = {}

    def __repr__(self):
        return "Model:\n" \
               "\tlayers: {0}\n" \
               "\tconnections: {1}\n".format(len(self.layers()),
                                             len(self.connections))

    def layers(self):
        return [t[0] for t in self._layers]

    def add_layer(self, layer, input_channel=None):
        """
        Add a GrFNN layer.

        Args:
            layer (:class:`.GrFNN`): the GrFNN to add to the model
            input_channel (`int` or `None`): If *None*, no external
                signal (stimulus) will be fed into this layer.
                Otherwise identifies the input channel to be fed into
                the layer.

        Raises:
            DuplicatedLayer
        """

        if layer not in self.layers():
            self._layers.append((layer, input_channel))
            self.connections[layer] = []    # list of connected layers.
                                            # List elems should be tuples
                                            # of the form (source_layer,
                                            # connextion_matrix)
            if layer.name == '':
                layer.name = 'Layer {}'.format(len(self._layers))


        else:
            raise DuplicatedLayer(layer)

    def connect_layers(self,
                       source,
                       destination,
                       matrix,
                       connection_type,
                       weight=1.0,
                       learn=None,
                       self_connect=True):
        """
        Connect two layers.

        Args:
            source (:class:`.GrFNN`): source layer (connections will be
                made from this layer to *destination*)
            destination (:class:`.GrFNN`): destination layer
                (connections will be made from *source* layer to this
                layer)
            matrix (:class:`numpy.array`): connection matrix
            connection_type (string): type of connection (e.g. '1freq', '2freq',
                '3freq', 'allfreq', 'all2freq')
            weight (float): connection weight factor.
            learn (:class:`.Cparmas`): Learning parameters. Is `None`, no
                learning will be performed
            self_connect (bool): whether or not to connect oscillators of the
                same frequency

        Returns:
            :class:`.Connection`: connection object created

        """
        if source not in self.layers():
            raise UnknownLayer(source)

        if destination not in self.layers():
            raise UnknownLayer(destination)

        conn = Connection(source, destination, matrix, connection_type,
                          weight=weight, learn_params=learn, self_connect=self_connect)
        self.connections[destination].append(conn)

        return conn

    def run(self, signal, t, dt, learn=False):
        """Run the model for a given stimulus, using "intertwined" RK4

        Args:
            signal (:class:`np.array_like`): external stimulus. If
                multichannel, the first dimension indexes time and the
                second one indexes channels
            t (:class:`np.array_like`): time vector corresponding to the
                signal
            dt (float): sampling period of `signal`
            learn (bool): enable connection learning

        Note:
            Intertwined means that a singe RK4 step needs to be run for
            all layers in the model, before running the next RK4 step.
            This is due to the fact that :math:`\\dot{z} = f(t, x(t), z(t))`.
            The "problem" is that :math:`x(t)` is also a function of
            :math:`z(t)`, so it needs to be updated for each layer in
            each RK step.


            Pseudo-code: ::

                for (i, stim) in stimulus:

                    for L in layers:
                        compute L.k1 given stim(-1), layers.z(-1)

                    for L in layers:
                        compute L.k2 given stim(-.5), layers.z(-1), L.k1

                    for L in layers:
                        compute L.k3 given stim(-.5), layers.z(-1), L.k2

                    for L in layers:
                        compute L.x(0), L.k4 given stim(0), layers.z(-1), L.k3

                    for L in layers:
                        compute L.z given L.k1, L.k2, L.k3, L.k4
                        L.TF[:,i] = L.z


        Note:
            The current implementation assumes **constant sampling
            period** ``dt``

        Note:
            If ``learn is True``, then the Hebbian Learning algorithm
            described in

               Edward W. Large. *Music Tonality, Neural Resonance and
               Hebbian Learning.* **Proceedings of the Third International
               Conference on Mathematics and Computation in Music
               (MCM 2011)**, pp. 115--125, Paris, France, 2011.

            is used to update the connections:

            .. math::

                \\dot{c_{ij}} = -\\delta_{ij}c_{ij} + k_{ij}
                                \\frac{z_{i}}{1-\\sqrt{\\varepsilon}z_i}
                                \\frac{\\bar{z}_{j}}
                                {1-\\sqrt{\\varepsilon}\\bar{z}_j}


        Warning:

            The above equation differs from the equation presented in
            the afore mentioned reference (:math:`j` was used as
            subscript in the last fractional term, instead of :math:`i`,
            as is in the paper). It seems there it was a typo in the
            reference, but this **must be confirmed** with the author.

            Furthermore, the current implementation assumes that :math:`i`
            indexes the source layer and :math:`j` indexes the destination
            layer (this needs confirmation as well).


        """

        num_frames = signal.shape[0]

        if signal.ndim == 1:
            signal = np.atleast_2d(signal).T

        # 1. prepare all the layers
        for L, inchan in self._layers:
            # FIXME
            L.TF = np.zeros((L.f.size, num_frames+1), dtype=COMPLEX)
            L.TF[:,0] = L.z

        # 2. Run "intertwined" RK4
        nc = len(str(num_frames))
        msg = '\r{{0:0{0}d}}/{1}'.format(nc, num_frames)
        for i in range(num_frames):

            #print "Frame {}:".format(i)
            s = signal[i, :]

            if i != num_frames-1:
                # linear interpolation should be fine
                x_stim = [s, 0.5*(s+signal[i+1, :]), signal[i+1, :]]
            else:
                x_stim = [s, s, s]

            # k1
            for L, inchan in self._layers:
                #print "Layer {} - k1".format(L.name)
                stim = 0 if inchan is None else x_stim[0][inchan]
                L.k1 = rk_step(L, dt, self.connections, stim, '')
                #print "k1:"
                #print L.k1*dt

            # k2
            for L, inchan in self._layers:
                #print "Layer {} - k2".format(L.name)
                stim = 0 if inchan is None else x_stim[1][inchan]
                L.k2 = rk_step(L, dt, self.connections, stim, 'k1')
                #print "k2:"
                #print L.k2*dt

            # k3
            for L, inchan in self._layers:
                #print "Layer {} - k3".format(L.name)
                stim = 0 if inchan is None else x_stim[1][inchan]
                L.k3 = rk_step(L, dt, self.connections, stim, 'k2')
                #print "k3:"
                #print L.k3*dt

            # k4
            for L, inchan in self._layers:
                #print "Layer {} - k4".format(L.name)
                stim = 0 if inchan is None else x_stim[2][inchan]
                L.k4 = rk_step(L, dt, self.connections, stim, 'k3')
                #print "k4:"
                #print L.k4*dt

            # final RK step
            for L in self.layers():
                L.z += dt*(L.k1 + 2.0*L.k2 + 2.0*L.k3 + L.k4)/6.0
                L.TF[:, i+1] = L.z
                #print "Layer {} z:".format(L.name)
                #print L.z

                # dispatch update event
                model_update_event.send(sender=L, z=L.z, t=t[0]+i*dt)

            # learn connections
            for L in self.layers():
                for j, conn in enumerate(self.connections[L]):
                    if conn.cparams is not None:
                        conn.matrix += learn_step(conn)
                        if not conn.self_connection:
                            # FIXME: This only works if source.f == destination.f
                            conn.matrix[range(len(conn.source.f)),
                                        range(len(conn.destination.f))] = 0

            # progress indicator
            sys.stdout.write(msg.format(i+1))
            sys.stdout.flush()

        sys.stdout.write(" done!\n")


# helper function that performs a single RK4 step (any of them)
def rk_step(layer, dt, connections, stim, pstep):
    """Single RK4 step

    Args:
        layer (:class:`grfnn.GrFNN`): layer to be integrated
        dt (float): integration step (in seconds)
        connections (dict): connection dictionary (see :class:`Model`)
        stim (float): external stimulus sample
        pstep (string): string identifying the previous RK step
            ``{'', 'k1', 'k2', 'k3'}``
    """
    h = dt if pstep is 'k3' else 0.5*dt
    k = getattr(layer, pstep, 0)
    z = layer.z + h*k
    conns = [None]*len(connections[layer])
    for i, c in enumerate(connections[layer]):
        src = c.source
        ks = getattr(src, pstep, 0)
        conns[i] = (src.z + h*ks, c)
    x = compute_input(layer, z, conns, stim)
    return layer.zdot(x, z, layer.f, layer.zparams)


# helper function that updates connection matrix
def learn_step(conn):
    """Update connection matrices

    Args:
        conn (:class:`.Connection`): connection object

    Returns:
        (:class:`np.ndarray`): derivative of connections (to use in
            connection update rule)
    """
    # TODO: test
    e = conn.destination.zparams.e
    zi = conn.source.z
    zj_conj = np.conj(conn.destination.z)  # verify which one should
                                           # be conjugated
    active = np.outer(zi*nl(zi, np.sqrt(e)),
                      zj_conj*nl(zj_conj, np.sqrt(e)))
    return -conn.d*conn.matrix + conn.k*active


