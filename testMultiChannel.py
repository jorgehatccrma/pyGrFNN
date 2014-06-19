"""Test muti-channel external stimulus"""

from pygrfnn.network import Model, make_connections
from pygrfnn.oscillator import Zparam
from pygrfnn.grfnn import GrFNN
from pygrfnn.vis import plot_connections
from pygrfnn.utils import timed

import numpy as np
import matplotlib.pyplot as plt


# load onset signal
odf1 = np.loadtxt('sampleOnsets/rumba.onset.txt')
odf2 = np.loadtxt('sampleOnsets/bossa.onset.txt')
fs_odf = odf1[0]
odf1 = odf1[1:]*0.5   # this factor is relevant (but annoying)
odf2 = odf2[1:]*0.0   # this factor is relevant (but annoying)

mt = max(odf1.size, odf2.size)
odf1 = np.hstack((odf1, np.zeros(mt-odf1.size)))
odf2 = np.hstack((odf2, np.zeros(mt-odf2.size)))

odf = np.vstack((odf1, odf2)).T

t_odf = np.arange(0, mt)
t_odf = t_odf/fs_odf


# create 3 GrFNNs (to visible and 1 hidden)
paramsVis = Zparam(0, -1.0, -0.25, 0, 0, .5)
paramsHid = Zparam(0.03, 1.0, -1.0, 0, 0, .5)

center_freq = 2.0
half_range =  2.5
oscs_per_octave = 64

layerV1 = GrFNN(paramsVis,
                fc=center_freq,
                octaves_per_side=half_range,
                oscs_per_octave=oscs_per_octave)

layerV2 = GrFNN(paramsVis,
                fc=center_freq,
                octaves_per_side=half_range,
                oscs_per_octave=oscs_per_octave)

layerH = GrFNN(paramsHid,
               fc=center_freq,
               octaves_per_side=half_range,
               oscs_per_octave=oscs_per_octave)


# define connectivities

# internal connectivity
rels = [1./3., 1./2., 1., 2., 3.]
internal_connsV1 = make_connections(layerV1, layerV1, 0.6, 0.1, harmonics=rels,
                                    complex_kernel=True, self_connect=False)

internal_connsV2 = make_connections(layerV2, layerV2, 0.6, 0.1, harmonics=rels,
                                    complex_kernel=True, self_connect=False)

internal_connsH = make_connections(layerH, layerH, 0.5, 0.1, harmonics=rels,
                                   complex_kernel=True, self_connect=False)


# inter layer connectivity
affConn1 = make_connections(layerV1, layerH, 0.75, 0.1, harmonics=[1],
                            complex_kernel=False, self_connect=True)

affConn2 = make_connections(layerV2, layerH, 0.75, 0.1, harmonics=[1],
                            complex_kernel=False, self_connect=True)

effConn1 = make_connections(layerH, layerV1, -0.75, .4, harmonics=[1],
                            complex_kernel=True, self_connect=False)

effConn2 = make_connections(layerH, layerV2, -0.75, .4, harmonics=[1],
                            complex_kernel=True, self_connect=False)

# create the model
net = Model()
net.add_layer(layerV1, input_channel=0)
net.add_layer(layerV2, input_channel=1)
net.add_layer(layerH, input_channel=None)

# add connectivity
internal_connsV1 = net.connect_layers(layerV1, layerV1, internal_connsV1)
internal_connsV2 = net.connect_layers(layerV2, layerV2, internal_connsV2)
internal_connsH = net.connect_layers(layerH, layerH, internal_connsH)
affConn1 = net.connect_layers(layerV1, layerH, affConn1)
affConn2 = net.connect_layers(layerV2, layerH, affConn2)
effConn1 = net.connect_layers(layerH, layerV1, effConn1)
effConn2 = net.connect_layers(layerH, layerV2, effConn2)


# run the model
@timed
def run_model():
  net.run(odf, t_odf, 1.0/fs_odf)

run_model()


# visualize outputs
TF = layerH.TF
f = layerH.f
T = 1.0/f

plot_onset_signal = False
plot_conns = True
plot_tf_output = True

if plot_onset_signal:
    import matplotlib.pyplot as plt
    plt.plot(t_odf, odf)
    plt.show()

if plot_conns:
    from pygrfnn.vis import plot_connections
    plot_connections(effConn1)
    plt.show()

if plot_tf_output:
    # from pygrfnn.vis import tf_simple
    # tf_simple(TF, t_odf, T, odf, np.abs)
    # tf_simple(TF, t_odf, T, None, np.abs)
    from pygrfnn.vis import tf_detail
    tf_detail(TF, t_odf, T, np.max(t_odf), odf, np.abs)
    plt.show()

