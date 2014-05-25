"""
Python implementation of the Gradient Frequency Neural Network model proposed by

Edward W. Large, Felix V. Almonte, and Marc J. Velasco.
A canonical model for gradient frequency neural networks.
Physica D: Nonlinear Phenomena, 239(12):905-911, 2010.
"""

__version_info__ = ('0', '0', '1')
__version__ = '.'.join(__version_info__)


from oscillator import *
from gfnn import *
from network import *
from defines import *
from utils import *
from vis import *