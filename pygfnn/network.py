import numpy as np
from utils import gaussPDF


def make_connections(source_f, dest_f, relations=np.array([1]), stdev=0.5):
    """
    Create a connection matrix from source to destination.

    :param source_f: ordered array of source frequencies
    :type source_f: numpy array of floats
    :param dest_f: ordered array of destination frequencies
    :type dest_f: numpy array of floats
    :param relations: frequency relations to connect (e.g. [1/3, 1/2, 1, 2, 3])
    :type relations: numpy array of floats
    :param stdev: standard deviation to use in the connections (to "spread" them with neighbors)
    """

    rows = source_f.size
    cols = dest_f.size

    # matrix of connections
    # connection matrices index source in rows and destination in columns. That is,
    # conn(i,j) specifies the connection weight from the i-th element to the j-th element
    conns = np.eye(rows, cols, dtype=np.float64)

    # Make self connections using a gaussian distribution
    # TODO: parametrize sigma and strength
    # TODO: optimize
    for row in range(rows):
        rf = dest_f/source_f[row]
        conns[row,] = gaussPDF(rf, 1, stdev/12.0)  # FIXME: what/why this x/12 factor? It was in the matlab code, but I don't get it (seems to relate to pitches, but then this is not the place!)

    return conns