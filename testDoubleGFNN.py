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
params1 = osc.Zparam(0, -1.0, -0.25, 0, 0, 1.0)
params2 = osc.Zparam(0.3, 1.0, -1.0, 0, 0, 1.0)


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



# # Temporary hack
# seed = odf[0]
# odf = odf[1:]
# odf = np.insert(odf, -1, 0)
# seed_x = layer.compute_input(layer1.z, [], seed)
# layer1.reset(x_1=seed_x)

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
    # tf_simple(TF, t_odf, T, None, np.abs)
    tf_detail(TF, t_odf, T, np.max(t_odf), odf, np.abs)


