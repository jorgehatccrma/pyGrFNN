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

import numpy as np
from utils import normalPDF
from utils import normalCDF
from utils import nl
from defines import COMPLEX, PI, PI_2
import sys


def make_connections(source, dest, strength, stdev, harmonics=None,
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
        stdev (float): standard deviation to use in the connections (to
            "spread" them with neighbors). Its units is *octaves* (e.g.
            stdev = 1.0 implies that roughly 68% of the weight will be
            distributed in two octaves---+/- 1 stdev---around the mean)
        harmonics (:class:`numpy.array`): frequency harmonics to connect
            (e.g. [1/3, 1/2, 1, 2, 3]). If *None*, it will be set to
            ``[1]``
        complex_kernel (bool): If *True*, the connections will be
            complex (i.e. include phase information). Otherwise, the
            connections will be real-valued weights.
        self_connect (bool): if *False*, the connection from source_f[i]
            to dest_f[j] (where source_f[i] == dest_f[j]) will be set to
            0

    ToDo:
        Revise the units of ``stdev``

    Returns:
        :class:`numpy.array`: Connection matrix (rows index destination and
            columns index source). In other words, to obtain the state at the
            destination, you must use `M.dot(source.z)`, where `M` is the
            connection matrix.

    """

    # matrix (2D arrray) of relative frequencies
    # source is indexed in columns and destination in rows. That is,
    # RF(i,j) specifies the relative frequency of source_f[j] w.r.t.
    # dest_f[i]
    [FS, FT] = np.meshgrid(source.f, dest.f)
    RF = FT/FS
    # RF = (dest.f/source.f.reshape(source.f.size, 1)).T

    assert RF.shape == (len(dest.f), len(source.f))

    # matrix of connections
    # connection matrices index source in rows and destination in
    # columns. That is, conn(i,j) specifies the connection weight from
    # the i-th element to the j-th element
    conns = np.zeros(RF.shape, dtype=COMPLEX)

    if harmonics is None:
        harmonics = [1]

    # Make self connections using a Gaussian distribution
    for h in harmonics:
        # TODO: verify if this is correct in the new Toolbox
        R = normalPDF(np.log2(RF), np.log2(h), stdev) / source.oscs_per_octave

        if complex_kernel:
            # TODO: verify if this is correct in the new Toolbox
            Q = PI_2*(2.0*normalCDF(np.log2(RF), np.log2(h), stdev)-1)
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
    """Convenient connection object

    Args:
        source (:class:`.GrFNN`): source layer
        destination (:class:`.GrFNN`): destination layer
        matrix (:class:`np.ndarray`): connection matrix
        learn_params (:class:`.Cparam`): learning params. No learning is performed
            when set to `None`
        self_connection (bool): if ``False``, the diagonal of the
            matrix is kept to 0 (even when learning is enabled)
        conn_type (string): type of GrFNN connections to use. Possible values:
        'allfreq', 'all2freq', '1freq', '2freq', '3freq'

    Attributes:
        source: :class:`.GrFNN` -- source layer
        destination: :class:`.GrFNN` -- destination layer
        matrix: :class:`np.ndarray` -- connection matrix
        cparams: :class:`.Cparam` -- Learning params (`None` means no learning)
        d: ``float`` -- "passive" learning rate (i.e. forgetting factor)
        k: ``float`` -- "active" learning rate

    """

    def __init__(self,
                 src,
                 dest,
                 matrix,
                 conn_type,
                 learn_params=None,
                 self_connection=True):
        self.source = src
        self.destination = dest
        self.matrix = matrix
        self.cparams = learn_params
        self.self_connection = self_connection
        self.conn_type = conn_type

    def __repr__(self):
        return "Connection:\n" \
               "\tsource:       {0}\n" \
               "\tdest.:        {1}\n" \
               "\tlearn_params: {2}\n" \
               "\tself_connect: {3}\n".format(self.source,
                                              self.destination,
                                              self.cparams,
                                              self.self_connection)


class Model(object):
    """
    A network of GrFNNs

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

    def add_layer(self, layer, input_channel=0):
        """Add a GrFNN layer.

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

        else:
            raise DuplicatedLayer(layer)

    def connect_layers(self,
                       source,
                       destination,
                       matrix,
                       connection_type,
                       learn=None):
        """Connect two layers.

        Args:
            source (:class:`.GrFNN`): source layer (connections will be
                made from this layer to *destination*)
            destination (:class:`.GrFNN`): destination layer
                (connections will be made from *source* layer to this
                layer)
            matrix (:class:`numpy.array`): Matrix of connection weights
            connection_type (string): type of connection (e.g. '1freq', '2freq',
                '3freq', 'allfreq', 'all2freq')
            learn (:class:`.Cparmas`): Learning parameters. Is `None`, no
                learning will be performed

        Returns:
            :class:`.Connection`: connection object created

        """

        # TODO: add sanity check?
        # TODO: add another method (or use duck typing) to pass harmonics or
        #       connection_type in matrix

        if source not in self.layers():
            raise UnknownLayer(source)

        if destination not in self.layers():
            raise UnknownLayer(destination)

        conn = Connection(source, destination, matrix, connection_type, learn)
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

        # helper function that performs a single RK4 step (any of them)
        def rk_step(layer, stim, pstep):
            """Single RK4 step

            Args:
                layer (:class:`grfnn.GrFNN`): layer to be integrated
                stim (float): external stimulus sample
                pstep (string): string identifying the previous RK step
                    ``{'', 'k1', 'k2', 'k3'}``
            """
            # print("***"+pstep)
            # print(layer)
            h = dt if pstep is 'k3' else 0.5*dt
            k = getattr(layer, pstep, 0)
            z = layer.z + h*k
            conns = [None]*len(self.connections[layer])
            for i, c in enumerate(self.connections[layer]):
                src = c.source
                ks = getattr(src, pstep, 0)
                conns[i] = (src.z + h*ks, c.matrix, c.conn_type)
            x = layer.compute_input(z, conns, stim)
            return layer.zdot(x, z)

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
            # if np.isnan(active).any():
            #     import pdb
            #     pdb.set_trace()
            return -conn.d*conn.matrix + conn.k*active

        num_frames = signal.shape[0]

        if signal.ndim == 1:
            signal = np.atleast_2d(signal).T

        # 1. prepare all the layers
        for L, inchan in self._layers:
            # FIXME
            L.TF = np.zeros((L.f.size, num_frames), dtype=COMPLEX)

        # 2. Run "intertwined" RK4
        nc = len(str(num_frames))
        msg = '\r{{0:0{0}d}}/{1}'.format(nc, num_frames)
        for i in range(num_frames):

            s = signal[i, :]

            if i != num_frames-1:
                # linear interpolation should be fine
                x_stim = [s, 0.5*(s+signal[i+1, :]), signal[i+1, :]]
            else:
                x_stim = [s, s, s]

            # k1
            for L, inchan in self._layers:
                stim = 0 if inchan is None else x_stim[0][inchan]
                L.k1 = rk_step(L, stim, '')

            # k2
            for L, inchan in self._layers:
                stim = 0 if inchan is None else x_stim[1][inchan]
                L.k2 = rk_step(L, stim, 'k1')

            # k3
            for L, inchan in self._layers:
                stim = 0 if inchan is None else x_stim[1][inchan]
                L.k3 = rk_step(L, stim, 'k2')

            # k4
            for L, inchan in self._layers:
                stim = 0 if inchan is None else x_stim[2][inchan]
                L.k4 = rk_step(L, stim, 'k3')

            # final RK step
            for L in self.layers():
                L.z += dt*(L.k1 + 2.0*L.k2 + 2.0*L.k3 + L.k4)/6.0
                L.TF[:, i] = L.z

            # learn connections
            for L in self.layers():
                for j, conn in enumerate(self.connections[L]):
                    if conn.cparams is not None:
                        # print np.isnan(conn.matrix).any()
                        conn.matrix += learn_step(conn)
                        if not conn.self_connection:
                            # FIXME: This only works if source.f == destination.f
                            conn.matrix[range(len(conn.source.f)),
                                        range(len(conn.destination.f))] = 0

            # progress indicator
            sys.stdout.write(msg.format(i+1))
            sys.stdout.flush()
        sys.stdout.write(" done!\n")
