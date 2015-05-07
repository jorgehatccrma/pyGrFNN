"""
Rhythm processing model
"""

from __future__ import division

from time import time
import sys
sys.path.append('../')  # needed to run the examples from within the package folder

import numpy as np
from scipy.signal import hilbert


from scipy.io import loadmat

from pygrfnn.network import Model, make_connections, modelFromJSON
from pygrfnn.oscillator import Zparam
from pygrfnn.grfnn import GrFNN

import matplotlib.pyplot as plt
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
                         reps=6.0,
                         lead_silence=0.0,
                         sr=sr,
                         click_freq=1200.0,
                         with_beat=False,
                         beat_freq=1800.0,
                         accented=False)
        x = Signal(x, sr=sr)
        s = onset_detection_signal(x)

        rms = np.sqrt(np.sum(s**2)/len(s))
        s *= 0.06/rms
        s = Signal(hilbert(s), sr=s.sr)
    t = s.time_vector()
    dt = 1/s.sr
    # print "SR: ", s.sr
    return s, t, dt


def rhythm_model():
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

    return modelFromJSON(rhythm_model_definition)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        pattern_name = sys.argv[1]
    else:
        pattern_name = "iso"

    s, t, dt = get_stimulus(pattern_name)

    model = rhythm_model()

    layer1, layer2 = model.layers()

    # Simulation
    if RT_display:
        plt.ion()
        plt.plot(t, s);
        plt.title('Stimulus')
        GrFNN_RT_plot(layer1, update_interval=2.0/s.sr, title='First Layer')
        GrFNN_RT_plot(layer2, update_interval=2.0/s.sr, title='Second Layer')

    tic = time()
    model.run(s, t, dt)
    print "Run time: {:0.1f} seconds".format(time() - tic)

    TF = layer2.Z
    r = np.sum(TF, 0)
    rms = np.sqrt(np.sum(r*np.conj(r))/len(r))
    r *= 0.06/rms
    plt.figure()
    plt.plot(t, np.real(r))
    plt.plot(t, np.real(s))

