"""
Rhythm processing model
"""

from __future__ import division

import sys
sys.path.append('../')  # needed to run the examples from within the package folder

import numpy as np
import matplotlib.pyplot as plt

from pygrfnn.network import Model, make_connections
from pygrfnn.oscillator import Zparam
from pygrfnn.grfnn import GrFNN
from pygrfnn.vis import plot_connections
from pygrfnn.vis import tf_detail
from pygrfnn.vis import GrFNN_RT_plot

from pyrhythm.library import get_pattern
from daspy import Signal
from daspy.processing import onset_detection_signal

p = get_pattern("iso")
sr = 22050
x, _ = p.as_signal(tempo=120.0,
                   reps=12.0,
                   lead_silence=0.0,
                   sr=sr,
                   click_freq=1200.0,
                   with_beat=False,
                   beat_freq=1800.0,
                   accented=False)

x = Signal(x, sr=sr)
s = onset_detection_signal(x)
s = s.normalize()
s *= 0.25
t = s.time_vector()
dt = 1/s.sr

print "SR: ", s.sr


zp1 = Zparam(0.00001, 0, 0, -2.0, 0, 1)
zp2 = Zparam(-0.4, 1.75, 0, -1.25, 0, 1)
# w = .4

# lambda =  -1; mu1 = 4; mu2 = -2.2; ceps = 1; kappa = 1; % Critical


# layers
# n1 = networkMake(1, 'hopf', alpha1, beta11, beta12, 0,  0,  neps1, ...
#                     'log', .375, 12, 321, ...
#                     'display', 2, 'save', 1, 'channel', 1);
# n2 = networkMake(2, 'hopf', alpha2, beta21, beta22,  0, 0, neps2, ...
#                     'log', .375, 12, 321, ...
#                     'display', 2, 'save', 1, 'channel', 1);
# n1.w = 3.0;


# Layers
# layer1 = GrFNN(zp1, frequency_range=(.375,12), num_oscs=321)
# layer2 = GrFNN(zp2, frequency_range=(.375,12), num_oscs=321)
layer1 = GrFNN(zp1, frequency_range=(.5,4), num_oscs=200)
layer2 = GrFNN(zp2, frequency_range=(.5,4), num_oscs=200)
layer1.w = 3.0

# Model
model = Model()
model.add_layer(layer1, input_channel=0)
model.add_layer(layer2)


# Connections
modes =      [1/3, 1/2, 1/1, 2/1, 3/1]
amps  =      [1,   1,   1,   1,   1 ]

# C1 = connectMake(n1, n1, 'gaus',  1, 1.05, 0, 1, 'modes', modes, 'amps', amps, 'sds', sds);
# C0 = connectMake(n1, n1, 'gaus',  1, 1.05, 0, 0, 'modes', modes, 'amps', amps, 'sds', sds);

C11 = make_connections(layer1, layer1, 1.0,  0.0015, modes=modes, mode_amps=amps)
C12 = make_connections(layer1, layer2, 1.0,  0.0015, modes=modes, mode_amps=amps)
C22 = make_connections(layer2, layer2, 1.0,  0.0015, modes=modes, mode_amps=amps)
C21 = make_connections(layer2, layer1, 1.0,  0.0015, modes=modes, mode_amps=amps)

# %% Connections
# % Internal
# n1 = connectAdd(n1, n1, C0, 'weight', .10, 'type', '2freq');

# % Afferent
# n2 = connectAdd(n1, n2, C1, 'weight', .40, 'type', '2freq');

# % Internal
# n2 = connectAdd(n2, n2, C0, 'weight', .10, 'type', '2freq');

# % Efferent
# n1 = connectAdd(n2, n1, C1, 'weight', .05, 'type', '2freq');

c11 = model.connect_layers(layer1, layer1, C11, '2freq', weight=.10)
c12 = model.connect_layers(layer1, layer2, C12, '2freq', weight=.40)
c22 = model.connect_layers(layer2, layer2, C22, '2freq', weight=.10)
c21 = model.connect_layers(layer2, layer1, C21, '2freq', weight=.05)


plt.ion()

GrFNN_RT_plot(layer1, update_interval=0.25, fig_name='First Layer', title='First Layer')
GrFNN_RT_plot(layer2, update_interval=0.25, fig_name='Second Layer', title='Second Layer')

model.run(s, t, dt)