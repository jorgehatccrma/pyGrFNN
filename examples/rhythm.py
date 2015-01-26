"""
Rhythm processing model
"""

from __future__ import division

import sys
sys.path.append('../')  # needed to run the examples from within the package folder

import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

from scipy.io import loadmat

from pygrfnn.network import Model, make_connections
from pygrfnn.oscillator import Zparam
from pygrfnn.grfnn import GrFNN
from pygrfnn.vis import plot_connections
from pygrfnn.vis import tf_detail
from pygrfnn.vis import GrFNN_RT_plot

from pyrhythm.library import get_pattern
from daspy import Signal
from daspy.processing import onset_detection_signal

# p = get_pattern("iso")
# sr = 22050
# x, _ = p.as_signal(tempo=120.0,
#                    reps=24.0,
#                    lead_silence=0.0,
#                    sr=sr,
#                    click_freq=1200.0,
#                    with_beat=False,
#                    beat_freq=1800.0,
#                    accented=False)

# x = Signal(x, sr=sr)
# s = onset_detection_signal(x)


# rms = np.sqrt(np.sum(s**2)/len(s))
# s *= 0.05/rms
# s = Signal(hilbert(s), sr=s.sr)


D = loadmat('examples/iso44_signal')

sr = float(D['Fs'][0][0])
s = D['signal'][0]  # analytical signal (generated in matlab)

s = Signal(s, sr=sr)

t = s.time_vector()
dt = 1/s.sr

print "SR: ", s.sr



zp1 = Zparam(0.00001, 0, -2.0, 0, 0, 1)
zp2 = Zparam(-0.4, 1.75, -1.25, 0, 0, 1)
# w = .4
# lambda =  -1; mu1 = 4; mu2 = -2.2; ceps = 1; kappa = 1; % Critical

# Layers
layer1 = GrFNN(zp1, frequency_range=(.375, 12), num_oscs=321, stimulus_conn_type='active')
layer2 = GrFNN(zp2, frequency_range=(.375, 12), num_oscs=321)
layer1.w = 3.0

# Model
model = Model()
model.add_layer(layer1, input_channel=0)
model.add_layer(layer2)


# Connections
modes = [1/3, 1/2, 1/1, 2/1, 3/1]
amps  = [1,   1,   1,   1,   1  ]

C11 = make_connections(layer1, layer1, 1.0,  1.05, modes=modes, mode_amps=amps)
C12 = make_connections(layer1, layer2, 1.0,  1.05, modes=modes, mode_amps=amps)
C22 = make_connections(layer2, layer2, 1.0,  1.05, modes=modes, mode_amps=amps)
C21 = make_connections(layer2, layer1, 1.0,  1.05, modes=modes, mode_amps=amps)

c11 = model.connect_layers(layer1, layer1, C11, '2freq', weight=.10, self_connect=False)
c12 = model.connect_layers(layer1, layer2, C12, '2freq', weight=.40)
c22 = model.connect_layers(layer2, layer2, C22, '2freq', weight=.10, self_connect=False)
c21 = model.connect_layers(layer2, layer1, C21, '2freq', weight=.05)

# Simulation
plt.ion()
GrFNN_RT_plot(layer1, update_interval=0.1, title='First Layer')
GrFNN_RT_plot(layer2, update_interval=0.1, title='Second Layer')
# GrFNN_RT_plot(layer1, update_interval=2.0/s.sr, title='First Layer')
# GrFNN_RT_plot(layer2, update_interval=2.0/s.sr, title='Second Layer')

model.run(s, t, dt)