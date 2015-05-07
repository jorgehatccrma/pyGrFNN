"""Network of GrFNNs

This module provides the necessary code to build a model connecting
multiple GrFNNs. Connections can be made within a GrFNN or between pairs
of them.

More importantly, it provides a method to run the model (process an
stimulus), including the ability to learn the connections between
GrFNNs.

To Do:
    - Re-implement learning

"""

import sys
import json
import warnings

import numpy as np
from scipy.stats import norm

import logging
logger = logging.getLogger('pygrfnn.network')

from pygrfnn.utils import nl
from pygrfnn.utils import fareyratio
from pygrfnn.defines import COMPLEX, PI, PI_2
from pygrfnn.grfnn import GrFNN
from pygrfnn.grfnn import compute_input
from pygrfnn.grfnn import grfnn_update_event
from pygrfnn.oscillator import Zparam
from pygrfnn.resonances import threeFreqMonomials


def make_connections(source, dest, strength=1.0, range=1.02,
                     modes=None, mode_amps=None,
                     complex_kernel=False, self_connect=True, **kwargs):
    """Creates a connection matrix, that connects source layer to destination
    layer.

    Args:
        source (:class:`.GrFNN`): source GrFNN (connections will be made
            between ``source`` and ``dest``)
        dest (:class:`.GrFNN`): destination GrFNN (connections will be
            made between ``source`` and ``dest``)
        strength (``float``): connection strength (multiplicative real
            factor)
        range (``float``): defines the standard deviation to use in the connections
            ("spread" them with neighbors). It is expressed as a ratio, to
            account for the log scale of the oscillators' frequency
        modes (:class:`numpy.ndarray`): frequency modes to connect
            (e.g. [1/3, 1/2, 1, 2, 3]). If ``None``, it will be set to
            ``[1]``
        mode_amplitudes (:class:`numpy.ndarray`): amplitude for each mode in
            ``modes`` (e.g. [.5, .75, 1, .75, .5]). If ``None``, it will be set to
            ``[1] * len(modes)``
        complex_kernel (``bool``): If ``True``, the connections will be
            complex (i.e. include phase information). Otherwise, the
            connections will be real-valued weights.
        self_connect (``bool``): if ``False``, the connection from ``source.f[i]``
            to ``dest.f[j]`` (where ``source_f[i] == dest_f[j]``) will be set to
            0

    Returns:
        :class:`numpy.ndarray`: Connection matrix (rows index destination and
            columns index source). In other words, to obtain the state at the
            destination, you must use ``C.dot(source.z)``, where ``C`` is the
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
        logger.info('No modes received. Setting single mode 1:1')

    if mode_amps is None:
        mode_amps = [1.0] * len(modes)
        logger.info('No mode amplitudes received. '
                    'Setting all modes to amplitude of 1')

    assert len(modes) == len(mode_amps)

    sigmas = np.abs(np.log2(range))*np.ones(len(modes))
    log_modes = np.log2(modes)

    per = np.floor(len(source.f)/(np.log2(source.f[-1])-np.log2(source.f[0])))
    df = 1.0/per

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

    logger.info(("Created connection matrix between"
                     " '{}' and '{}' ({}x{})").format(source.name,
                                                      dest.name,
                                                      *conns.shape))

    return strength * conns


class DuplicatedLayer(Exception):
    """
    Raised when attempting to add a previously added layer to a network

    Attributes:
        layer (:class:`.GrFNN`): duplicated layer
    """

    def __init__(self, layer):
        self.layer = layer


class UnknownLayer(Exception):
    """
    Raised when attempting to use a layer unknown to the network

    Attributes:
        layer (:class:`.GrFNN`): unknown layer
    """

    def __init__(self, layer):
        self.layer = layer

    def __str__(self):
        return "Unknown layer %s. Did you forget to call " \
               "'add_layer(layer)'?" % (repr(self.layer))


class Cparam(object):
    """Convenience class to encapsulate connectivity learning parameters.

    Attributes:
        l: (``float``): linear forgetting rate :math:`\\lambda`
        m1: (``float``):  non-linear forgetting rate 1 :math:`\\mu_1`
        m2: (``float``): non-linear forgetting rate 2 :math:`\\mu_2`
        k: (``float``): learning rate :math:`\\kappa`
        e: (``float``): Coupling strength :math:`\\varepsilon`

    ToDo:
        Revise this (learning is probably broke, as I haven't updated it in a
        while)

    Note:
        This class is analogous to :class:`.Zparam`

    """

    def __init__(self, lmbda=-1.0, mu1=0.0, mu2=0.0,
                 kappa=1.0, epsilon=1.0):
        """Constructor.

        Args:
            lmbda (``float``): :math:`\\lambda` (defaults to: -1.0) (**this is
                not a typo: `lambda` is a keyword in python, so we used a
                misspelled version of the word**)
            mu1 (``float``): :math:`\\mu_1` (defaults to: -1.0)
            mu2 (``float``): :math:`\\mu_2` (defaults to: -0.25)
            kappa (``float``): :math:`\\kappa` (defaults to: 0.0)
            epsilon (``float``): :math:`\\varepsilon` (defaults to: 1.0)

        """
        logger.warning('Apparently you want to use learning (plasticity). '
                       'This feature is probably not working properly '
                       '(if at all). YOU HAVE BEEN WARNED!')
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
        matrix (:class:`numpy.ndarray`): connection matrix
        conn_type (``string``): type of GrFNN connections to use. Possible values:
            ``allfreq``, ``all2freq``, ``1freq``, ``2freq``, ``3freq``
        self_connect (``bool``): if ``False``, the diagonal of the
            matrix is kept to 0 (even when learning is enabled)
        weight (``float``): frequency weight factor
        learn_params (:class:`.Cparam`): learning params. No learning is performed
            when set to ``None``.

    Attributes:
        source (:class:`.GrFNN`): source layer
        destination (:class:`.GrFNN`): destination layer
        matrix (:class:`numpy.ndarray`): connection matrix
        cparams (:class:`.Cparam`): Learning params (`None` means no learning)
        self_connect (``bool``): If ``False``, the connection weights connecting
            two oscillators of the same frequency will be set to 0
        RF (:class:`numpy.ndarray`): array of frequency ratio.
            ``RF[i,j] = dest.f[i]/source.f[j]``
        farey_num (:class:`numpy.ndarray`): Farey numerator for the
            frequency relationship ``RF[i,j]`` (only set one using ``2freq``
            coupling).
        farey_den (:class:`numpy.ndarray`): Farey denominator for the
            frequency relationship ``RF[i,j]`` (only set one using ``2freq``
            coupling).
        monomials (``list``): List of :class:`python.namedtuples` of the from
            ``(indices, exponents)``. There is one ``namedtuple`` for each
            oscillator in ``destination`` GrFNN. Each tuple is formed by of two
            :class:`numpy.ndarray` with indices and exponents of a 3-freq
            monomial, each one of size Nx3, where N is the number of monomials
            associated to the corresponding oscillator in ``destination``. This
            attribute is only set for ``3freq`` connection type.

    Warning:
        Learning has not been updated lately, so it is probably broken.

    Warning:
        ``1freq`` connectivity is not implemented
    """

    def __init__(self,
                 source,
                 destination,
                 matrix,
                 conn_type,
                 self_connect,
                 weight=1.0,
                 learn_params=None,
                 conn_params=None):
        self.source = source
        self.destination = destination
        self.matrix = matrix.copy()
        self.cparams = learn_params
        self.self_connect = self_connect
        self.conn_type = conn_type

        # this is only for 'log' spaced GrFNNs
        self.weights = weight * destination.f

        [FS, FT] = np.meshgrid(self.source.f, self.destination.f)
        self.RF = FT/FS

        self.farey_num, self.farey_den = None, None
        self.monomials = None

        # if not self.self_connect:
        #     self.matrix[np.logical_and(self.farey_num==1, self.farey_den==1)] = 0
        if not self.self_connect:
            self.matrix[self.RF==1.0] = 0


        if conn_type == '2freq':
            # compute integer relationships between frequencies of both layers
            tol = 0.05
            if conn_params is not None:
                if 'tol' in connection_params:
                    tol = connection_params['tol']
            logger.info('Setting up 2-freq coupling between '
                         '"{}" and "{}", using tol={}'.format(source.name,
                                                              destination.name,
                                                              tol))
            self.farey_num, self.farey_den, _, _ = fareyratio(self.RF, tol)

        elif conn_type == '3freq':
            # default params
            N = 3
            tol = 5e-3
            lowest_order_only = True

            if conn_params is not None:
                if 'N' in conn_params:
                    N = conn_params['N']
                if 'tol' in conn_params:
                    tol = conn_params['tol']
                if 'lowest_order_only' in conn_params:
                    lowest_order_only = conn_params['lowest_order_only']

            logger.info('Setting up 3-freq coupling between '
                        '"{}" and "{}", with params N={}, '
                        'tol = {} and '
                        'lowest_order_only = {}'.format(source.name,
                                                        destination.name,
                                                        N,
                                                        tol,
                                                        lowest_order_only))

            self.monomials = threeFreqMonomials(self.source.f,
                                                self.destination.f,
                                                self_connect,
                                                N=N,
                                                tol=tol,
                                                lowest_order_only=lowest_order_only)

            def avg():
                a = 0
                for i, m in enumerate(self.monomials):
                    a += m.indices.shape[0]
                    # logger.debug("{}: {}".format(i, m.indices.shape))
                return a/len(self.monomials)
            logger.info('Found {} monomials per oscillator (avg)'.format(avg()))


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
        connections (``dict``): dictionary of
            connections. *Keys* are destination layers (:class:`.GrFNN`) and
            *values* are a list of connections (:class:`.Connection`).

    """

    def __init__(self, name=""):
        """
        **Model constructor**

        Args:
            name (``string``): (optional) model name (empty by default). This
                argument is required when defining a model from a dictionary or
                JSON object (see :func:`.modelFromJSON`)
        """
        self.name = name

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
        """
        Return a list of GrFNNs in the model.
        """
        return [t[0] for t in self._layers]

    def add_layer(self, layer, input_channel=None):
        """
        Add a GrFNN (layer) to the model

        Args:
            layer (:class:`.GrFNN`): the GrFNN to add to the model
            input_channel (`int` or `None`): If ``None`` (default), no external
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
                       self_connect=False,
                       connection_params=None):
        """
        Connect two layers in a :class:`Model`.

        Args:
            source (:class:`.GrFNN`): source layer (connections will be
                made from this layer to *destination*)
            destination (:class:`.GrFNN`): destination layer
                (connections will be made from *source* layer to this
                layer)
            matrix (:class:`numpy.array`): connection matrix
            connection_type (``string``): type of connection (e.g. ``1freq``,
                ``2freq``, ``3freq``, ``allfreq``, ``all2freq``)
            weight (``float``): connection weight factor.
            learn (:class:`.Cparams`): Learning parameters. Is ``None``, no
                learning will occur.
            self_connect (``bool``): whether or not to connect oscillators of the
                same frequency (defaults to ``False``).
            connection_params (``dict``): dictionary with connection_type
                specific parameters

        Returns:
            :class:`.Connection`: connection object created

        Warning:
            ``1freq`` connection is not implemented.

        """
        if source not in self.layers():
            raise UnknownLayer(source)

        if destination not in self.layers():
            raise UnknownLayer(destination)

        conn = Connection(source, destination, matrix, connection_type,
                          weight=weight, learn_params=learn,
                          self_connect=self_connect,
                          conn_params=connection_params)
        self.connections[destination].append(conn)

        return conn

    def run(self, signal, t, dt, learn=False):
        """Run the model for a given stimulus, using "intertwined" RK4

        Args:
            signal (:class:`numpy.ndarray`): external stimulus. If
                multichannel, the first dimension indexes time and the
                second one indexes channels
            t (:class:`numpy.ndarray`): time vector corresponding to the
                signal
            dt (``float``): sampling period of ``signal``
            learn (``bool``): enable connection learning

        Warning:
            Learning has not been updated lately, so it is probably broken.


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
            reference, but this **must be verified** with the author.

            Furthermore, the current implementation assumes that :math:`i`
            indexes the source layer and :math:`j` indexes the destination
            layer (this needs confirmation as well).


        """
        num_frames = signal.shape[0]

        if signal.ndim == 1:
            signal = np.atleast_2d(signal).T

        # dispatch update even for the initial state
        for L in self.layers():
            if L.save_states:
                L._prepare_Z(len(t))
                logger.info('"{}" ready to store TFR'.format(L.name))
            grfnn_update_event.send(sender=L, t=t[0], index=0)

        # Run fixed step RK4
        nc = len(str(num_frames))
        msg = '\r{{0:0{0}d}}/{1}'.format(nc, num_frames)
        for i in range(1, num_frames):

            s = signal[i-1, :]
            s_next = signal[i, :]
            # input signal (need interpolated values for k2 & k3)
            # linear interpolation should be fine
            x_stim = [s, 0.5*(s+s_next), 0.5*(s+s_next), s_next]

            for k in range(4):
                for L, inchan in self._layers:
                    stim = 0 if inchan is None else x_stim[k][inchan]
                    # ToDo DRY: The following if/elif... could be further
                    # simplified to adhere to DRY, but for some reason the
                    # version implemented using setattr() didn't give the same
                    # results?!
                    if k==0:
                        L.k1 = _rk_step(L, dt, self.connections, stim, 'k0')
                    elif k==1:
                        L.k2 = _rk_step(L, dt, self.connections, stim, 'k1')
                    elif k==2:
                        L.k3 = _rk_step(L, dt, self.connections, stim, 'k2')
                    elif k==3:
                        L.k4 = _rk_step(L, dt, self.connections, stim, 'k3')


            # final RK step
            for L in self.layers():
                L.z += dt*(L.k1 + 2.0*L.k2 + 2.0*L.k3 + L.k4)/6.0
                # dispatch update event
                grfnn_update_event.send(sender=L, t=t[i], index=i)

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

        # force final update (mainly to make sure the last state is
        # reflected in on-line displays)
        for L in self.layers():
            grfnn_update_event.send(sender=L,
                                    t=t[num_frames-1],
                                    index=num_frames-1,
                                    force=True)


# helper function that performs a single RK4 step (any of them)
def _rk_step(layer, dt, connections, stim, pstep):
    """Single RK4 step

    Args:
        layer (:class:`grfnn.GrFNN`): layer to be integrated
        dt (``float``): integration step (in seconds)
        connections (``dict``): connection dictionary (see :class:`Model`)
        stim (``float``): external stimulus sample
        pstep (``string``): string identifying the previous RK step
            ``{'', 'k1', 'k2', 'k3'}``

    Returns:
        :class:`numpy.ndarray`: :math:`\\dot{z}` updated for the input
            :class:`.GrFNN` (``layer``).
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
# FIXME: this needs to be updated! It hasn't been touched after getting a hold
# to Matlab's GrFNN Toolbox
def learn_step(conn):
    """Update connection matrices

    Args:
        conn (:class:`.Connection`): connection object

    Returns:
        :class:`np.ndarray`: derivative of connections (to use in
            connection update rule)

    ToDo:
        Revise (learning needs to be updated)

    """
    # TODO: test
    e = conn.destination.zparams.e
    zi = conn.source.z
    zj_conj = np.conj(conn.destination.z)  # verify which one should
                                           # be conjugated
    active = np.outer(zi*nl(zi, np.sqrt(e)),
                      zj_conj*nl(zj_conj, np.sqrt(e)))
    return -conn.d*conn.matrix + conn.k*active



def modelFromJSON(definition=None):
    """
    Utility function to create a model (Network) from a JSON object (or a python
    ``dict``)

    Args:
        definition (``string``, ``JSON`` or ``dict``): model definition

    Returns:
        :class:`.Model`: Model, as specified in the input object.
    """
    try:
        D = json.loads(definition)
    except TypeError:
        # assume we received a dict (already parsed JSON)
        D = dict(definition)
    except Exception:
        raise

    model = Model(name=D["name"])
    layers = dict()
    # create layers
    for L in D["layers"]:
        # print "Creating layer", L["name"]

        layer = GrFNN(**L)
        if "input_channel" in L:
            model.add_layer(layer, input_channel=L["input_channel"])
        else:
            model.add_layer(layer)

        layers[layer.name] = layer

    # connect layers
    for C in D["connections"]:
        try:
            # print "Creating connection {} -> {}".format(C["source_name"], C["target_name"])
            source = layers[C["source_name"]]
            target = layers[C["target_name"]]


            M = make_connections(source, target, **C)

            conn_params = dict()
            for key in ('weight', 'learn', 'self_connect', 'connection_params'):
                if key in C:
                    conn_params[key] = C[key]
            c = model.connect_layers(source, target, matrix=M,
                                     connection_type=C['connection_type'],
                                     **conn_params)
        except Exception as e:
            print "Error creating connection {}".format(C)
            print e

    return model



if __name__ == '__main__':

    rhythm_model_definition = """
    {
        "name": "Sensory Motor Rhythm model",
        "layers": [
            {
                "name": "sensory network",
                "zparams": {
                    "alpha": 0.00001,
                    "beta1": 0.0,
                    "beta2": -2.0,
                    "delta1": 0.0,
                    "delta2": 0.0,
                    "epsilon": 1.0
                },
                "frequency_range": [0.375, 12.0],
                "num_oscs": 321,
                "stimulus_conn_type": "linear",
                "w": 3.0,
                "input_channel": 0
            },
            {
                "name": "motor network",
                "zparams": {
                    "alpha": -0.4,
                    "beta1": 1.75,
                    "beta2": -1.25,
                    "delta1": 0.0,
                    "delta2": 0.0,
                    "epsilon": 1.0
                },
                "frequency_range": [0.375, 12.0],
                "num_oscs": 321,
                "stimulus_conn_type": "active"
            }
        ],
        "connections": [
            {
                "source_name": "sensory network",
                "target_name": "sensory network",
                "modes": [0.333333333333, 0.5, 1, 2.0, 3.0],
                "amps": [1, 1, 1, 1, 1],
                "strength": 1.0,
                "range": 1.05,
                "connection_type": "2freq",
                "self_connect": false,
                "weight": 0.1
            },
            {
                "source_name": "sensory network",
                "target_name": "motor network",
                "modes": [0.333333333333, 0.5, 1, 2.0, 3.0],
                "amps": [1, 1, 1, 1, 1],
                "strength": 1.25,
                "range": 1.05,
                "connection_type": "2freq",
                "self_connect": true,
                "weight": 0.4
            },
            {
                "source_name": "motor network",
                "target_name": "motor network",
                "modes": [0.333333333333, 0.5, 1, 2.0, 3.0],
                "amps": [1, 1, 1, 1, 1],
                "strength": 1.0,
                "range": 1.05,
                "connection_type": "2freq",
                "self_connect": false,
                "weight": 0.1
            },
            {
                "source_name": "motor network",
                "target_name": "sensory network",
                "modes": [0.333333333333, 0.5, 1, 2.0, 3.0],
                "amps": [1, 1, 1, 1, 1],
                "strength": 0.2,
                "range": 1.05,
                "connection_type": "2freq",
                "self_connect": true,
                "weight": 0.05
            }
        ]
    }
    """

    rhythmModel = modelFromJSON(rhythm_model_definition)
