"""Network of GFNNs related code
"""

import numpy as np
from utils import normalPDF
from utils import normalCDF
from utils import f, nml
from defines import COMPLEX, FLOAT, PI, PI_2


def make_connections(source, dest, harmonics=np.array([1]), stdev=0.5, complex_kernel=False, self_connect=True, conn_type='rhythm'):
    """Creates a connection matrix from source to destination.

    Args:
        source (:class:`.GFNN`): source GFNN (connections will be made between this and *dest*)
        dest (:class:`.GFNN`): destination GFNN (connections will be made between *source* and *this*)
        harmonics (:class:`numpy.array`): frequency harmonics to connect (e.g. [1/3, 1/2, 1, 2, 3])
        stdev (float): standard deviation to use in the connections (to "spread" them with neighbors)
        complex_kernel (bool): If *True*, the connections will be complex (i.e. include phase information).
            Otherwise, the connections will be real-valued weights.
        self_connect (bool): if *False*, the connection from source_f[i] to dest_f[j]
            (where source_f[i] == dest_f[j]) will be set to 0
        conn_type (string): type of connections. Possible values are 'rhythm' (default) or 'gauss'

    Returns:
        :class:`numpy.array`: Connection matrix (rows index source and columns index destination)

    """

    # matrix (2D arrray) of relative frequencies
    # source is indexed in rows and destination in columns. That is,
    # RF(i,j) specifies the relative frequency of dest_f[j] w.r.t. source_f[i]
    RF = dest.f/source.f.reshape(source.f.size, 1)

    # matrix of connections
    # connection matrices index source in rows and destination in columns. That is,
    # conn(i,j) specifies the connection weight from the i-th element to the j-th element
    conns = np.zeros(RF.shape, dtype=COMPLEX)

    # gauss implies a 1:1 relation only (I think)
    if conn_type is 'gauss':
        harmonics = [1]

    # Make self connections using a Gaussian distribution
    # TODO: optimize
    for h in harmonics:
        R = normalPDF(np.log2(RF), np.log2(h), stdev/12.0)/source.oscs_per_octave
                                            # FIXME: what/why this x/12 factor? It was in the matlab code,
                                            # but I don't get it (seems to relate to pitches, but then
                                            # this is not the place!)
        # Also, in the original implementation, R was divided by the number of oscillators per octave.
        # Why? I think it should be either be divided by cumsum(R(row,:)) [preserve energy] or
        # max(R(row,:)) [full self-feedback]

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

        conns = conns + R * np.exp(1j*Q) # This whole complex kernel business seems odd


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
        return "Unknown layer %s. Maybe you forgot to call 'add_layer(layer)'?" % (repr(self.layer))



class Model(object):
    """
    A network of GFNNs

    Different GFNNs are referred to as layers. Layers can be added as visible or hidden; the former
    means that it will directly receive external stimulus, while the later implies that the inputs will
    consist only of internal connections (internal to the layer or from other layers in the network).

    Attributes:
        visible_layers: list of GFNN layers that will receive the external signal
        hidden_layers: list of GFNN layers that won't receive the external signal
        connections: dictionary of connection matrices. Keys correspond to destination layers.
            Values are tuples identifying the source layer and connection matrix. In other words
            ``connections[layerTo] = (layerFrom, conn_matrix)``

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
            visible (bool): If *True*, the external signal (stimulus) will be fed into this layer

        Raises:
            DuplicatedLayer: see :class:`.DuplicatedLayer`
        """

        # TODO: add sanity check

        if layer not in self.visible_layers + self.hidden_layers:
            if visible:
                self.visible_layers.append(layer)
            else:
                self.hidden_layers.append(layer)

            self.connections[layer] = []    # list of connected layers. List elements
                                            # should be tuples of the form
                                            # (source_layer, connextion_matrix)

        else:
            raise DuplicatedLayer(layer)



    def connect_layers(self, source, destination, matrix):
        """Connect two layers.

        Args:
            source (:class:`.GFNN`): source layer (connections will be made from this layer to *destination*)
            destination (:class:`.GFNN`): destination layer (connections will be made from *source* layer to this layer)
            matrix (:class:`numpy.array`): Matrix of connection weights
        """

        # TODO: add sanity check?
        # TODO: add another method (or use duck typing) to pass harmonics or connection_type in matrix

        if source not in self.visible_layers+self.hidden_layers:
            raise UnknownLayer(source)

        if destination not in self.visible_layers+self.hidden_layers:
            raise UnknownLayer(destination)

        self.connections[destination].append((source, matrix))



    # def process_signal(self, signal, t, dt):
    #     """Compute the TF representation of an input signal

    #     Note:
    #         TODO: raise exception when shapes of **signal** and **t** mismatch?

    #     Args:
    #         signal (numpy complex array): input signal (stimulus)
    #         t (numpy float array): time vector (must have the same shape as *signal*)
    #         dt (float): input signal's sample period (inverse of the sample rate)

    #     """

    #     # 1. prepare all the layers
    #     for layer in self.visible_layers + self.hidden_layers:
    #         layer.TF = np.zeros((layer.f.size, signal.size), dtype=COMPLEX)
    #         # layer.x = np.zeros((layer.f.size, signal.size), dtype=COMPLEX)

    #     # 2. run it one sample at a time
    #     for (i, x_stim) in enumerate(signal):
    #         # 1. compute the inputs for all layers
    #         input_processed = []
    #         for layer in self.visible_layers:
    #             x = self.compute_input(layer, self.connections[layer], x_stim)
    #             input_processed.append((layer, x))
    #         for layer in self.hidden_layers:
    #             x = self.compute_input(layer, self.connections[layer])
    #             input_processed.append((layer, x))
    #             # print layer, np.sum(x)

    #         # 2. "run" all the layers
    #         for layer, x in input_processed:
    #             layer.process_time_step(dt, x)
    #             # layer.x[:,i] = x
    #             layer.TF[:,i] = layer.z


    def solve_for_stimulus(self, signal, t, dt):
        """Run the model, using "intertwined" RK4

        Note:
            Assumes constant *dt*

        Pseudo-code: ::

            for (i, x_stim) in stimulus:

                for layer in layers:
                    compute layer.k1 given layer.x(-1), layers.z(-1)

                for layer in layers:
                    compute layer.k2 given x_stim(-0.5), layers.z(-1), layers.k1

                for layer in layers:
                    compute layer.k3 given x_stim(-0.5), layers.z(-1), layers.k2

                for layer in layers:
                    compute layer.x(0), layer.z(0), layer.k4 given x_stim(0), layers.z(-1), layer.k3
                    layer.TF[:,i] = layer.z(0)


        """

        all_layers = self.visible_layers + self.hidden_layers

        # 1. prepare all the layers
        for layer in all_layers:
            layer.TF = np.zeros((layer.f.size, signal.size), dtype=COMPLEX)

        # RK4 reminder

        # xh = 0.5*(x+x_1)   # for now, linear interpolation
        # dth = 0.5*dt
        # k1 = diffeq(x_1, z_1)
        # k2 = diffeq(xh,  z_1 + dth*k1)
        # k3 = diffeq(xh,  z_1 + dth*k2)
        # k4 = diffeq(x,   z_1 + dt*k3)
        # return z_1 + dt*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0


        # TODO: there's a lot of repetition in the following code. A helper
        #       inline function could help

        # 2. Run "intertwined" RK4
        dth = 0.5 * dt
        for (i, s) in enumerate(signal):

            if s != signal[-1]:
                # linear interpolation should be fine
                x_stim = [s, 0.5*(s+signal[i+1]), signal[i+1]]
            else:
                x_stim = [s, s, s]

            # import pdb
            # pdb.set_trace()

            # k1
            for layer in all_layers:
                stim = 0
                if layer in self.visible_layers:
                    stim = x_stim[0]
                conns = [(L.z, M) for (L, M) in self.connections[layer]]
                x_1 = layer.compute_input(layer.z, conns, stim)
                layer.k1 = layer.dzdt(x_1, layer.z)


            # k2
            for layer in all_layers:
                stim = 0
                if layer in self.visible_layers:
                    stim = x_stim[1]

                z = layer.z + dth*layer.k1
                conns = [(L.z + dth*L.k1, M) for (L, M) in self.connections[layer]]
                x_halfway = layer.compute_input(z, conns, stim)
                layer.k2 = layer.dzdt(x_halfway, z)

            # k3
            for layer in all_layers:
                stim = 0
                if layer in self.visible_layers:
                    stim = x_stim[1]

                z = layer.z + dth*layer.k2
                conns = [(L.z + dth*L.k2, M) for (L, M) in self.connections[layer]]
                x_halfway = layer.compute_input(z, conns, stim)
                layer.k3 = layer.dzdt(x_halfway, z)

            # k4
            for layer in all_layers:
                stim = 0
                if layer in self.visible_layers:
                    stim = x_stim[2]

                z = layer.z + dt*layer.k3
                conns = [(L.z + dth*L.k3, M) for (L, M) in self.connections[layer]]
                x = layer.compute_input(z, conns, stim)
                layer.k4 = layer.dzdt(x, z)

                # final RK step
                layer.z = layer.z + dt*(layer.k1 + 2.0*layer.k2 + 2.0*layer.k3 + layer.k4)/6.0
                layer.TF[:,i] = layer.z