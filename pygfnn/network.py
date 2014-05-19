import numpy as np
from utils import normalPDF
from utils import normalCDF
from utils import f, nml
from defines import COMPLEX, PI


def make_connections(source_f, dest_f, harmonics=np.array([1]), stdev=0.5, complex_kernel=False, self_connect=True):
    """
    Create a connection matrix from source to destination.

    :param source_f: ordered array of source frequencies
    :type source_f: numpy array of floats
    :param dest_f: ordered array of destination frequencies
    :type dest_f: numpy array of floats
    :param harmonics: frequency harmonics to connect (e.g. [1/3, 1/2, 1, 2, 3])
    :type harmonics: numpy array of floats
    :param stdev: standard deviation to use in the connections (to "spread" them with neighbors)
    :type stdev: float
    :param complex_kernel: If *True*, the connections will be complex (i.e. include phase information).
        Otherwise, the connections will be real-valued weights.
    :type complex_kernel: bool
    :param self_connect: if *False*, the connection from source_f[i] to dest_f[j]
        (where source_f[i] == dest_f[j]) will be set to 0
    :type self_connect: bool


    :return: Connection matrix. Rows index source and Columns index destination
    :rtype: numpy complex array (2D)
    """

    # matrix (2D arrray) of relative frequencies
    # source is indexed in rows and destination in columns. That is,
    # RF(i,j) specifies the relative frequency of dest_f[j] w.r.t. source_f[i]
    RF = dest_f/source_f.reshape(source_f.size, 1)

    # matrix of connections
    # connection matrices index source in rows and destination in columns. That is,
    # conn(i,j) specifies the connection weight from the i-th element to the j-th element
    conns = np.zeros(RF.shape, dtype=COMPLEX)


    # Make self connections using a Gaussian distribution
    # TODO: optimize
    for h in harmonics:
        # R = normalPDF(RF/h, 1, stdev/12.0)  # FIXME: what/why this x/12 factor? It was in the matlab code,
        #                                     # but I don't get it (seems to relate to pitches, but then
        #                                     # this is not the place!)

        # R = normalPDF(np.log2(RF/h), np.log2(1), stdev/12.0)  # FIXME: what/why this x/12 factor? It was in the matlab code,
        #                                     # but I don't get it (seems to relate to pitches, but then
        #                                     # this is not the place!)

        R = normalPDF(np.log2(RF), np.log2(h), stdev/12.0)  # FIXME: what/why this x/12 factor? It was in the matlab code,
                                            # but I don't get it (seems to relate to pitches, but then
                                            # this is not the place!)

        # import pdb
        # pdb.set_trace()

        if not self_connect:
            R[RF==1] = 0

        # In the original implementation, R was divided by the number of oscillators per octave.
        # Why? I think it should be either be divided by cumsum(R(row,:)) [preserve energy] or
        # max(R(row,:)) [full self-feedback]
        # For now I'll simply normalize the end result to get max(row)==1
        if complex_kernel:
            # TODO: implement
            # (from NLTFT:)
            # pi*(2*normcdf(log2(RF), log2(harms(nn)), log2(1+sd(nn)))-1);
            Q = PI*(2.0*normalCDF(np.log2(RF), np.log2(h), np.log2(1+stdev))-1);
            Q[R<=np.abs(conns)] = 0
            conns = conns + R * np.exp(1j*Q) # FIXME: This whole complex kernel business seems odd
        else:
            conns = conns + R

    # # normalization
    # # TODO: verify correctness
    # tmp = np.max(np.abs(conns.T),axis=0)
    # conns = (conns.T/tmp).T

    return conns


class DuplicatedLayer(Exception):
    """
    Raised when attempting to add a previously added layer to a network

    Attribute:
        layer -- duplicated layer
    """

    def __init__(self, layer):
        self.layer = layer


class UnknownLayer(Exception):
    """
    Raised when attempting to use a layer unknown to the network

    Attribute:
        layer -- unknown layer
    """

    def __init__(self, layer):
        self.layer = layer

    def __str__(self):
        return "Unknown layer %s. Maybe you forgot to call 'add_layer(layer)'?" % (repr(self.layer))



class Model(object):
    """
    A network of GFNNs. Different GFNNs will be referred to as layers.
    """

    def __init__(self):
        """
        TODO: describe stuff (specially self.connections)
        """

        #: Visible GFNN: list of GFNN layers that will receive the external signal
        self.visible_layers = []

        #: Hidden GFNNs: list of GFNN layers that won't receive the external signal
        self.hidden_layers = []

        # connections
        self.connections = {}

        pass


    def add_layer(self, layer, visible=True):
        """
        Add a GFNN layer.

        :param layer: the GFNN to add to the model
        :type layer: :class:`.GFNN`

        :param visible: If *True*, the external signal (stimulus) will be fed into this layer
        :type visible: bool

        :raises DuplicatedLayer: see :class:`.DuplicatedLayer`
        """

        # TODO: add sanity check

        if layer not in self.visible_layers + self.hidden_layers:
            if visible:
                self.visible_layers.append(layer)
            else:
                self.hidden_layers.append(layer)

            self.connections[layer] = []    # list of connected layers. List elements
                                            # should be tuples of the form
                                            # (destination_layer, connextion_matrix)

        else:
            raise DuplicatedLayer(layer)



    def connect_layers(self, source, destination, connections):
        """
        Connect two layers.

        :param source: Source layer (connections will be made from this layer to *destination*)
        :type source: :class:`.GFNN`
        :param destination: Destination layer (connections will be made from *source* layer to this layer)
        :type destination: :class:`.GFNN`
        :param connections: Matrix of connection weights
        :type connections: numpy complex array. It's shape must be (source.f.size, destination.f.size)
        """

        # TODO: add sanity check
        # TODO: add another method (or use duck typing) to pass harmonics or connection_type in connections

        if source not in self.visible_layers+self.hidden_layers:
            raise UnknownLayer(source)

        if destination not in self.visible_layers+self.hidden_layers:
            raise UnknownLayer(destination)

        self.connections[destination].append((source, connections))




    def process_signal(self, signal, t, dt):
        """
        Compute the TF representation of an input signal
        """

        def compute_input(layer, external_conns, x_stim=0):
            """
            external_conns is a list of tuples of the form (source_layer, connectioc_matrix)
            """
            # compute overall input (external signal + internal connections + eff/aff connections)
            # For reference: input pre-processing from NLTFT
            # x = f(n.e, x_stim) + f(n.e, nml(x_aff)) + f(n.e, nml(x_int)) + f(n.e, nml(x_eff));

            # process external signal (stimulus)
            x = f(x_stim, layer.zparams.e)
            if layer.internal_conns is not None:
                # process internal signal (via internal connections)
                x_int = layer.z.dot(layer.internal_conns)
                x = x + f(nml(x_int), layer.zparams.e)
            # process other external inputs (afferent / efferent)
            for (source, conns) in external_conns:
                x_ext = source.z.dot(conns)
                # print np.sum(x_ext)
                x = x + f(nml(x_ext), layer.zparams.e)
                # print x_ext
            return x

        # 1. reset / prepare all the layers
        for layer in self.visible_layers + self.hidden_layers:
            layer.reset()
            layer.TF = np.zeros((layer.f.size, signal.size), dtype=COMPLEX)

        # 2. run it one sample at a time
        for (i, x_stim) in enumerate(signal):
            # 1. compute the inputs for all layers
            input_processed = []
            for layer in self.visible_layers:
                x = compute_input(layer, self.connections[layer], x_stim)
                input_processed.append((layer, x))
            for layer in self.hidden_layers:
                x = compute_input(layer, self.connections[layer])
                input_processed.append((layer, x))
                # print layer, np.sum(x)

            # 2. "run" all the layers
            for layer, x in input_processed:
                layer.process_time_step(dt, x)
                layer.TF[:,i] = layer.z




