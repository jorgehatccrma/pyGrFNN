from pygfnn.network import Model
from pygfnn.oscillator import Zparam
from pygfnn.gfnn import GFNN

import numpy as np


# load onset signal
# odf = np.loadtxt('sampleOnsets/rumba.onset.txt')
odf = np.loadtxt('sampleOnsets/bossa.onset.txt')
fs_odf = odf[0]
odf = odf[1:]*0.5   # this factor is relevant (but annoying)
t_odf = np.arange(0, odf.size)
t_odf = t_odf/fs_odf


# create single GFNN
params = Zparam(0, -0.25, -0.5, 0, 0, 1)
layer = GFNN(params,
             fc=2,
             octaves_per_side=2,
             oscs_per_octave=64)

# add internal connectivity
rels = [1./3., 1./2., 1., 2., 3.]
layer.connect_internally(relations=rels,
                         internal_strength=0.6,
                         internal_stdev=0.5,
                         complex_kernel=True)

# create the model
model = Model()
model.add_layer(layer)

# run the model
model.run(odf, t_odf, 1.0/fs_odf)




# visualize results
TF = layer.TF
f = layer.f
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
    plot_connections(layer.internal_conns, layer.f, layer.f)

if plot_tf_output:
    # from pygfnn.vis import tf_simple
    # tf_simple(TF, t_odf, T, odf, np.abs)
    # tf_simple(TF, t_odf, T, None, np.abs)
    from pygfnn.vis import tf_detail
    tf_detail(TF, t_odf, T, np.max(t_odf), odf, np.abs)

