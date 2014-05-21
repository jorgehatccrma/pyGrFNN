import pygfnn.network as model
import pygfnn.oscillator as osc
import pygfnn.gfnn as gfnn

import numpy as np
import matplotlib.pyplot as plt



plot_onset_signal = False
plot_internal_conns = False
plot_tf_output = True


# load onset signal
# odf = np.loadtxt('sampleOnsets/bossa.onset.txt')
odf = np.loadtxt('sampleOnsets/rumba.onset.txt')
fs_odf = odf[0]
odf = odf[1:]
t_odf = np.arange(0, odf.size)
t_odf = t_odf/fs_odf


# create a pair of GFNNs
params1 = osc.Zparam(0, -1.0, -0.25, 0, 0, .5)
params2 = osc.Zparam(0.03, 1.0, -1.0, 0, 0, .5)


internal_strength = 0.6
internal_stdev = 0.5

center_freq = 2.0
half_range =  2
oscs_per_octave = 64

layer1 = gfnn.GFNN(params1,
                   fc=center_freq,
                   octaves_per_side=half_range,
                   oscs_per_octave=oscs_per_octave)

layer2 = gfnn.GFNN(params2,
                   fc=center_freq,
                   octaves_per_side=half_range,
                   oscs_per_octave=oscs_per_octave)


rels = [1./3., 1./2., 1., 2., 3.]
# rels = [1]
layer1.connect_internally(relations=rels,
                          internal_strength=internal_strength,
                          internal_stdev=internal_stdev,
                          complex_kernel=True)

layer2.connect_internally(relations=rels,
                          internal_strength=internal_strength,
                          internal_stdev=internal_stdev,
                          complex_kernel=True)

net = model.Model()
net.add_layer(layer1)
net.add_layer(layer2, visible=False)

affConnStrength = 0.75;
affConnStdDev = 0.5;
effConnStrength = -0.75;
effConnStdDev = 4.0;

affConn = affConnStrength * model.make_connections (layer1,
                                                    layer2,
                                                    harmonics=[1],
                                                    stdev=affConnStdDev,
                                                    complex_kernel=False,
                                                    self_connect=False,
                                                    conn_type='gauss')

effConn = effConnStrength * model.make_connections (layer2,
                                                    layer1,
                                                    harmonics=[1],
                                                    stdev=effConnStdDev,
                                                    complex_kernel=True,
                                                    self_connect=False,
                                                    conn_type='gauss')



# # using NLTFT matrices
# import scipy.io as io
# mats = io.loadmat('conns.mat')
# print "TRACE"

# import pdb
# pdb.set_trace()

# affConn = mats['affConn']
# effConn = mats['effConn']
# intConn = mats['internalpattern']
# layer1.internal_conns = intConn
# layer2.internal_conns = intConn




net.connect_layers(layer1, layer2, affConn)
net.connect_layers(layer2, layer1, effConn)




# run the model
net.solve_for_stimulus(odf*0.5, t_odf, 1.0/fs_odf)
f = layer1.f
T = 1.0/f
TF = layer2.TF



# PLOTS

if plot_onset_signal:
    plt.plot(t_odf, odf)
    plt.show()


if plot_internal_conns:
    from pygfnn.vis import plot_connections
    plot_connections(effConn, layer1.f, layer2.f)


if plot_tf_output:
    from pygfnn.vis import tf_detail
    # from pygfnn.vis import tf_simple
    # tf_simple(TF, t_odf, T, odf, np.abs)
    # tf_simple(TF[:,:700], t_odf[:700], T, odf[:700], np.abs)
    # tf_simple(TF, t_odf, T, None, np.abs)
    tf_detail(TF, t_odf, T, 5, odf, np.abs)


