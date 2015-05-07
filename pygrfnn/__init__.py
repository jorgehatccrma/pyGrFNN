#coding=utf-8
"""
Python implementation of the Gradient Frequency Neural Network model proposed by

Edward W. Large, Felix V. Almonte, and Marc J. Velasco.
A canonical model for gradient frequency neural networks.
Physica D: Nonlinear Phenomena, 239(12):905-911, 2010.

"""

__version_info__ = ('0', '1', '0')
__version__ = '.'.join(__version_info__)

import logging
pygrfnn_logger = logging.getLogger('pygrfnn')
pygrfnn_logger.addHandler(logging.NullHandler())


from grfnn import GrFNN
from oscillator import Zparam
from network import Model, make_connections

# from oscillator import *
# from grfnn import *
# from network import *
# from defines import *
# from utils import *
# from vis import *
# from moviegen import *