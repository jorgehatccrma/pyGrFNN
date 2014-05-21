import pygfnn.network as model
import pygfnn.oscillator as osc
import pygfnn.gfnn as gfnn

import numpy as np


# load onset signal
odf = np.loadtxt('sampleOnsets/rumba.onset.txt')
# odf = np.loadtxt('sampleOnsets/bossa.onset.txt')
fs_odf = odf[0]
odf = odf[1:]*0.5   # this factor is relevant (but annoying)
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


# add internal connectivities
rels = [1./3., 1./2., 1., 2., 3.]
layer1.connect_internally(relations=rels,
                          internal_strength=internal_strength,
                          internal_stdev=internal_stdev,
                          complex_kernel=True)

layer2.connect_internally(relations=rels,
                          internal_strength=internal_strength,
                          internal_stdev=internal_stdev,
                          complex_kernel=True)


# create the model
net = model.Model()
net.add_layer(layer1)
net.add_layer(layer2, visible=False)

# add inter-layer connections
affConnStrength = 0.75;
affConnStdDev = 0.5;
effConnStrength = -0.75;
effConnStdDev = 4.0;

affConn = affConnStrength * model.make_connections (layer1,
                                                    layer2,
                                                    harmonics=[1],
                                                    stdev=affConnStdDev,
                                                    complex_kernel=False,
                                                    self_connect=True,
                                                    conn_type='gauss')

effConn = effConnStrength * model.make_connections (layer2,
                                                    layer1,
                                                    harmonics=[1],
                                                    stdev=effConnStdDev,
                                                    complex_kernel=True,
                                                    self_connect=False,
                                                    conn_type='gauss')

net.connect_layers(layer1, layer2, affConn)
net.connect_layers(layer2, layer1, effConn)


# run the model
net.run(odf, t_odf, 1.0/fs_odf)



# visualize outputs
TF = layer2.TF
f = layer1.f
T = 1.0/f

plot_onset_signal = False
plot_internal_conns = False
plot_tf_output = True

if plot_onset_signal:
    import matplotlib.pyplot as plt
    plt.plot(t_odf, odf)
    plt.show()

if plot_internal_conns:
    from pygfnn.vis import plot_connections
    plot_connections(effConn, layer1.f, layer2.f)

if plot_tf_output:
    # from pygfnn.vis import tf_simple
    # tf_simple(TF, t_odf, T, odf, np.abs)
    # tf_simple(TF, t_odf, T, None, np.abs)
    from pygfnn.vis import tf_detail
    tf_detail(TF, t_odf, T, np.max(t_odf), odf, np.abs)


