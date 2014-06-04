"""Network of GFNNs related code

This module provides the necessary code to build a model connecting multiple
GFNNs. Connections can be made within a GFNN or between pairs of them.

More importantly, it provides a method to run the model.

To Dos:
    - Implement other types of connectivities
    - Implement learning?

"""

import numpy as np
from utils import normalPDF
from utils import normalCDF
from utils import nl
from defines import COMPLEX, PI, PI_2


def make_connections(source, dest, strength, stdev, harmonics=None,
                     complex_kernel=False, self_connect=True):
    """Creates a connection matrix from source to destination.

    Args:
        source (:class:`.GFNN`): source GFNN (connections will be made between
            this and *dest*)
        dest (:class:`.GFNN`): destination GFNN (connections will be made
            between *source* and *this*)
        strength (float): connection strength (multiplicative real factor)
        stdev (float): standard deviation to use in the connections (to "spread"
            them with neighbors). Its units is *octaves* (e.g. stdev = 1.0
            implies that roughly 68% of the weight will be distributed in two
            octaves---+/- 1 stdev---around the mean)
        harmonics (:class:`numpy.array`): frequency harmonics to connect
            (e.g. [1/3, 1/2, 1, 2, 3]). If *None*, it will be set to ``[1]``
        complex_kernel (bool): If *True*, the connections will be complex (i.e.
            include phase information). Otherwise, the connections will be
            real-valued weights.
        self_connect (bool): if *False*, the connection from source_f[i] to
            dest_f[j] (where source_f[i] == dest_f[j]) will be set to 0

    ToDo:
        Revise the units of ``stdev``

    Returns:
        :class:`numpy.array`: Connection matrix (rows index source and columns
            index destination)

    """

    # matrix (2D arrray) of relative frequencies
    # source is indexed in rows and destination in columns. That is,
    # RF(i,j) specifies the relative frequency of dest_f[j] w.r.t. source_f[i]
    RF = dest.f/source.f.reshape(source.f.size, 1)

    # matrix of connections
    # connection matrices index source in rows and destination in columns. That
    # is, conn(i,j) specifies the connection weight from the i-th element to the
    # j-th element
    conns = np.zeros(RF.shape, dtype=COMPLEX)

    if harmonics is None: harmonics = [1]

    # Make self connections using a Gaussian distribution
    for h in harmonics:
        # R = normalPDF(np.log2(RF), np.log2(h), stdev/12.0)/source.oscs_per_octave
        # # FIXME: what/why this x/12 factor? It was in the matlab code,
        # # but I don't get it (seems to relate to pitches, but then
        # # this is not the place!)

        # TODO: verify if this is correct in the new Toolbox
        R = normalPDF(np.log2(RF), np.log2(h), stdev)/source.oscs_per_octave

        if complex_kernel:
            # # From NLTFT
            # if conn_type is 'gauss':
            #     Q = PI_2*(2.0*normalCDF(np.log2(RF), np.log2(h), stdev/12.0)-1);
            # else:
            #     Q = PI*(2.0*normalCDF(np.log2(RF), np.log2(h), np.log2(1+stdev))-1);

            # TODO: verify if this is correct in the new Toolbox
            Q = PI_2*(2.0*normalCDF(np.log2(RF), np.log2(h), stdev)-1);
        else:
            Q = np.zeros(R.shape)

        if not self_connect:
            R[RF==1] = 0
            Q[RF==1] = 0

        conns = conns + R * np.exp(1j*Q)
        # This whole complex kernel business seems odd

    return strength * conns


class DuplicatedLayer(Exception):
    """
    Raised when attempting to add a previously added layer to a network

    Attribute:
        layer (:class:`.GFNN`): duplicated layer
    """

    def __init__(self, layer):
        self.layer = layer


class UnknownLayer(Exception):
    """
    Raised when attempting to use a layer unknown to the network

    Attribute:
        layer (:class:`.GFNN`): unknown layer
    """

    def __init__(self, layer):
        self.layer = layer

    def __str__(self):
        return "Unknown layer %s. Did you forget to call 'add_layer(layer)'?" % \
                (repr(self.layer))



class Model(object):
    """
    A network of GFNNs

    Different GFNNs are referred to as layers. Layers can be added as visible or
    hidden; the former means that it will directly receive external stimulus,
    while the later implies that the inputs will consist only of internal
    connections (internal to the layer or from other layers in the network).

    Attributes:
        visible_layers: list of GFNN layers that will receive the external signal
        hidden_layers: list of GFNN layers that won't receive the external signal
        connections: dictionary of connection matrices. Keys correspond to
            destination layers. Values are tuples identifying the source layer
            connection matrix and learning parameters: ``connections[layerTo] =
            (layerFrom, conn_matrix, learn, d, k)``

    """

    def __init__(self):
        """
        TODO: describe stuff (specially self.connections)
        """

        # Visible GFNN: list of GFNN layers that will receive the external signal
        self.visible_layers = []

        # Hidden GFNNs: list of GFNN layers that won't receive the external signal
        self.hidden_layers = []

        # connections
        self.connections = {}

        pass


    def add_layer(self, layer, visible=True):
        """Add a GFNN layer.

        Args:
            layer (:class:`.GFNN`): the GFNN to add to the model
            visible (bool): If *True*, the external signal (stimulus) will be
                fed into this layer

        Raises:
            DuplicatedLayer: see :class:`.DuplicatedLayer`
        """

        if layer not in self.visible_layers + self.hidden_layers:
            if visible:
                self.visible_layers.append(layer)
            else:
                self.hidden_layers.append(layer)

            self.connections[layer] = []    # list of connected layers. List
                                            # elems should be tuples of the form
                                            # (source_layer, connextion_matrix)

        else:
            raise DuplicatedLayer(layer)



    def connect_layers(self, source, destination, matrix, learn=False, d=0.5, k=0.5):
        """Connect two layers.

        Args:
            source (:class:`.GFNN`): source layer (connections will be made from
                this layer to *destination*)
            destination (:class:`.GFNN`): destination layer (connections will be
                made from *source* layer to this layer)
            matrix (:class:`numpy.array`): Matrix of connection weights
            learn (bool): if *True*, connections will be learned
            d (float): "passive" learning rate
            k (float): "active" learning rate
        """

        # TODO: add sanity check?
        # TODO: add another method (or use duck typing) to pass harmonics or
        #       connection_type in matrix

        if source not in self.visible_layers+self.hidden_layers:
            raise UnknownLayer(source)

        if destination not in self.visible_layers+self.hidden_layers:
            raise UnknownLayer(destination)

        self.connections[destination].append((source, matrix, learn, d, k))



    def run(self, signal, t, dt, learn=False):
        """Run the model for a given stimulus, using "intertwined" RK4

        Intertwined means that a singe RK4 step needs to be run for all layers
        in the model, before running the next RK4 step. This is due to the fact
        that :math:`\\dot{z} = f(t, x(t), z(t))`. The "problem" is that
        :math:`x(t)` is also a function of :math:`z(t)`.

        Args:
            signal (:class:`np.array_like`): external stimulus
            t (:class:`np.array_like`): time vector corresponding to the signal
            dt (float): sampling period of `signal`
            learn (bool): enable connection learning

        Pseudo-code: ::

            for (i, x_stim) in stimulus:

                for L in layers:
                    compute L.k1 given L.x(0), layers.z(-1)

                for L in layers:
                    compute L.k2 given x_stim(0.5), layers.z(-1), L.k1

                for L in layers:
                    compute L.k3 given x_stim(0.5), layers.z(-1), L.k2

                for L in layers:
                    compute L.x(0), L.k4 given x_stim(1), layers.z(-1), L.k3

                for L in layers:
                    compute L.z give L.k1, L.k2, L.k3, L.k4
                    L.TF[:,i] = L.z


        Note:
            The current implementation assumes **constant sampling period** ``dt``
        Note:

            If ``learn is True``, then the Hebbian Learning algorithm described in

               Edward W. Large. *Music Tonality, Neural Resonance and Hebbian Learning.*
               **Proceedings of the Third International Conference on Mathematics and
               Computation in Music (MCM 2011)**, pp. 115--125, Paris, France, 2011.

            is used to update the connections:

            .. math::

                \\dot{c_{ij}} = -\\delta_{ij}c_{ij} + k_{ij}
                                \\frac{z_{i}}{1-\\sqrt{\\varepsilon}z_i}
                                \\frac{\\bar{z}_{j}}{1-\\sqrt{\\varepsilon}\\bar{z}_j}


        Warning:

            The above equation differs from the equation presented in the
            mentioned reference (:math:`j` was used as subscript in the last
            fractional term, instead of :math:`i`, as is in the paper). It seems
            there it was a typo in the reference, but this **must be confirmed**
            with the author.

            Furthermore, the current implementation assumes that :math:`i`
            indexes the source layer and :math:`j` indexes the destination layer
            (this needs confirmation as well).


        """

        # helper function that performs a single RK4 step (any of them)
        def rk_step(layer, stim, step):
            """Single RK4 step

            Args:
                layer (:class:`gfnn.GFNN`): layer to be integrated
                stim (float): external stimulus sample
                step (string): string identifying the previous RK step
                    ``{'', 'k1', 'k2', 'k3'}``
            """
            h = dt if step is 'k3' else 0.5*dt
            z = layer.z + h*getattr(layer, step, 0)
            # this pythonic trick might be too unreadable
            conns = [(L.z+h*getattr(L, step, 0), M) for (L, M, __, __, __)
                                                    in self.connections[layer]]
            x = layer.compute_input(z, conns, stim)
            return layer.dzdt(x, z)


        # helper function that updates connection matrix
        def learn_step(conn, source, dest, d, k):
            """Update connection matrices

            Args:
                conn (:class:`np.ndarray`): connection matrix
                source (:class:`gfnn.GFNN`): origin of the connection (rows in
                    ``conn``)
                dest (:class:`gfnn.GFNN`): destination of the connection (cols
                    in ``conn``)
                d (float): "passive" learning rate :math:`\\delta`
                k (float): "active" learning rate :math:`\\k_{ij}`
            """
            # TODO: test
            e = dest.zparams.e
            zi = dest.z
            zj_conj = np.conj(source.z)  # verify which one should be conjugated
            active = np.outer( zi*nl(zi, np.sqrt(e)),
                               zj_conj*nl(zj_conj, np.sqrt(e)) )
            cdot = -d*conn + k*active
            conn += cdot


        # 1. prepare all the layers
        all_layers = self.visible_layers + self.hidden_layers
        for layer in all_layers:
            layer.TF = np.zeros((layer.f.size, signal.size), dtype=COMPLEX)


        # 2. Run "intertwined" RK4
        for (i, s) in enumerate(signal):

            if s != signal[-1]:
                # linear interpolation should be fine
                x_stim = [s, 0.5*(s+signal[i+1]), signal[i+1]]
            else:
                x_stim = [s, s, s]

            # k1
            for L in all_layers:
                stim = x_stim[0] if L in self.visible_layers else 0
                L.k1 = rk_step(L, stim, '')

            # k2
            for L in all_layers:
                stim = x_stim[1] if L in self.visible_layers else 0
                L.k2 = rk_step(L, stim, 'k1')

            # k3
            for L in all_layers:
                stim = x_stim[1] if L in self.visible_layers else 0
                L.k3 = rk_step(L, stim, 'k2')

            # k4
            for L in all_layers:
                stim = x_stim[2] if L in self.visible_layers else 0
                L.k4 = rk_step(L, stim, 'k3')

            # final RK step
            for L in all_layers:
                L.z = L.z + dt*(L.k1 + 2.0*L.k2 + 2.0*L.k3 + L.k4)/6.0
                L.TF[:,i] = L.z

            # # learn connections
            # if learn:
            #     for L in all_layers:
            #         conns, S, learn, d, k = self.connections[L]
            #         if learn:
            #             learn_step(conns, S, L, d=d, k=k)

