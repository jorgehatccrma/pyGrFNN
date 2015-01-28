"""
Rhythm processing model
"""

from __future__ import division

from time import time
import sys
sys.path.append('../')  # needed to run the examples from within the package folder

import numpy as np
from scipy.signal import hilbert

import vis
if vis.MPL:
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

use_matlab_stimulus = False
RT_display = True

def get_stimulus(pattern_name="iso"):
    if use_matlab_stimulus:
        D = loadmat('examples/iso44_signal')
        sr = float(D['Fs'][0][0])
        s = D['signal'][0]  # analytical signal (generated in matlab)
        s = Signal(s, sr=sr)
    else:
        p = get_pattern(pattern_name)
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

        rms = np.sqrt(np.sum(s**2)/len(s))
        s *= 0.05/rms
        s = Signal(hilbert(s), sr=s.sr)
    t = s.time_vector()
    dt = 1/s.sr
    # print "SR: ", s.sr
    return s, t, dt


# GrFNNs definition
zp1 = Zparam(0.00001, 0, -2.0, 0, 0, 1)
zp2 = Zparam(-0.4, 1.75, -1.25, 0, 0, 1)
# w = .4
# lambda =  -1; mu1 = 4; mu2 = -2.2; ceps = 1; kappa = 1; % Critical

# Layers
layer1 = GrFNN(zp1, frequency_range=(.375, 12), num_oscs=321, stimulus_conn_type='active')
layer2 = GrFNN(zp2, frequency_range=(.375, 12), num_oscs=321)
layer1.w = 3.0

# store layer's states
layer1.save_states = True
layer2.save_states = True


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

c11 = model.connect_layers(layer1, layer1, C11, '2freq', weight=.10)
c12 = model.connect_layers(layer1, layer2, C12, '2freq', weight=.40, self_connect=True)
c22 = model.connect_layers(layer2, layer2, C22, '2freq', weight=.10)
c21 = model.connect_layers(layer2, layer1, C21, '2freq', weight=.05, self_connect=True)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        pattern_name = sys.argv[1]
    else:
        pattern_name = "iso"

    s, t, dt = get_stimulus(pattern_name)

    # Simulation
    if vis.MPL and RT_display:
        plt.ion()
        plt.plot(t, s);
        plt.title('Stimulus')
        GrFNN_RT_plot(layer1, update_interval=2.0/s.sr, title='First Layer')
        GrFNN_RT_plot(layer2, update_interval=2.0/s.sr, title='Second Layer')

    tic = time()
    model.run(s, t, dt)
    print "Run time: {:0.1f} seconds".format(time() - tic)

    if vis.MPL:
        TF = layer2.Z
        r = np.sum(TF, 0)
        rms = np.sqrt(np.sum(r*np.conj(r))/len(r))
        r *= 0.05/rms
        plt.figure()
        plt.plot(t, np.real(r))
        plt.plot(t, np.real(s))

