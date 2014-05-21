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
from defines import COMPLEX, PI, PI_2


def make_connections(source, dest, harmonics=None, stdev=0.5,
                     complex_kernel=False, self_connect=True,
                     conn_type='rhythm'):
    """Creates a connection matrix from source to destination.

    Args:
        source (:class:`.GFNN`): source GFNN (connections will be made between
            this and *dest*)
        dest (:class:`.GFNN`): destination GFNN (connections will be made
            between *source* and *this*)
        harmonics (:class:`numpy.array`): frequency harmonics to connect
            (e.g. [1/3, 1/2, 1, 2, 3]). If *None*, it will be set to ``[1]``
        stdev (float): standard deviation to use in the connections (to "spread"
            them with neighbors)
        complex_kernel (bool): If *True*, the connections will be complex (i.e.
            include phase information). Otherwise, the connections will be
            real-valued weights.
        self_connect (bool): if *False*, the connection from source_f[i] to
            dest_f[j] (where source_f[i] == dest_f[j]) will be set to 0
        conn_type (string): type of connections. Possible values are 'rhythm'
            (default) or 'gauss'

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

    # gauss implies a 1:1 relation only (I think)
    if conn_type is 'gauss': harmonics = [1]
    if harmonics is None: harmonics = [1]

    # Make self connections using a Gaussian distribution
    for h in harmonics:
        R = normalPDF(np.log2(RF), np.log2(h), stdev/12.0)/source.oscs_per_octave
        # FIXME: what/why this x/12 factor? It was in the matlab code,
        # but I don't get it (seems to relate to pitches, but then
        # this is not the place!)
        # Also, in the original implementation, R was divided by the number of
        # oscillators per octave. Why? I think it should be either be divided by
        # cumsum(R(row,:)) [preserve energy] or max(R(row,:)) [full self-feedback]

        if complex_kernel:
            if conn_type is 'gauss':
                # From NLTFT:
                # Q = (pi/2)*(2*normcdf(log2(RF), mn, sd/12)-1);
                Q = PI_2*(2.0*normalCDF(np.log2(RF), np.log2(h), stdev/12.0)-1);
            else:
                # From NLTFT:
                # Q = pi*(2*normcdf(log2(RF), log2(harms(nn)), log2(1+sd(nn)))-1);
                Q = PI*(2.0*normalCDF(np.log2(RF), np.log2(h), np.log2(1+stdev))-1);
                # Q[R<=np.abs(conns)] = 0
        else:
            Q = np.zeros(R.shape)

        if not self_connect:
            R[RF==1] = 0
            Q[RF==1] = 0

        conns = conns + R * np.exp(1j*Q)
        # This whole complex kernel business seems odd

    return conns


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
            and connection matrix. In other words ``connections[layerTo] =
            (layerFrom, conn_matrix)``

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



    def connect_layers(self, source, destination, matrix):
        """Connect two layers.

        Args:
            source (:class:`.GFNN`): source layer (connections will be made from
                this layer to *destination*)
            destination (:class:`.GFNN`): destination layer (connections will be
                made from *source* layer to this layer)
            matrix (:class:`numpy.array`): Matrix of connection weights
        """

        # TODO: add sanity check?
        # TODO: add another method (or use duck typing) to pass harmonics or
        #       connection_type in matrix

        if source not in self.visible_layers+self.hidden_layers:
            raise UnknownLayer(source)

        if destination not in self.visible_layers+self.hidden_layers:
            raise UnknownLayer(destination)

        self.connections[destination].append((source, matrix))



    def run(self, signal, t, dt):
        """Run the model for a given stimulus, using "intertwined" RK4

        Intertwined means that a singe RK4 step needs to be run for all layers
        in the model, before running the next RK4 step. This is due to the fact
        that :math:`\\dot{z} = f(t, x(t), z(t))`. The "problem" is that
        :math:`x(t)` is also a function of :math:`z(t)`.

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
            The current implementation assumes constant *dt*

        """

        # helper function that performs a single RK4 step (any of them)
        def rk_step(layer, stim, step):
            h = dt if step is 'k3' else 0.5*dt
            z = layer.z + h*getattr(layer, step, 0)
            # this pythonic trick might be too unreadable
            conns = [(L.z+h*getattr(L, step, 0), M) for (L, M) in self.connections[layer]]
            x = layer.compute_input(z, conns, stim)
            return layer.dzdt(x, z)


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

