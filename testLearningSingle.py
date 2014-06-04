from pygfnn.network import Model, make_connections
from pygfnn.oscillator import Zparam
from pygfnn.gfnn import GFNN
from pygfnn.vis import plot_connections

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

# internal connectivity
internal_conns = make_connections(layer, layer, 0.0, 0.5,
                                  harmonics=[1./3., 1./2., 1., 2., 3.],
                                  complex_kernel=True,
                                  self_connect=False)


# create the model
model = Model()

# setup layers
model.add_layer(layer)

# add connectivity
conn = model.connect_layers(layer, layer, internal_conns, learn=True, d=0.001, k=0.001)

# run the model
model.run(odf, t_odf, 1.0/fs_odf)



# visualize results
TF = layer.TF
f = layer.f
T = 1.0/f

plot_onset_signal = False
plot_conns = True
plot_tf_output = True

if plot_onset_signal:
    import matplotlib.pyplot as plt
    plt.plot(t_odf, odf)
    plt.show()

if plot_conns:
    from pygfnn.vis import plot_connections
    plot_connections(conn, display_op=lambda x: np.log(np.abs(x)))

if plot_tf_output:
    # from pygfnn.vis import tf_simple
    # tf_simple(TF, t_odf, T, odf, np.abs)
    # tf_simple(TF, t_odf, T, None, np.abs)
    from pygfnn.vis import tf_detail
    tf_detail(TF, t_odf, T, np.max(t_odf), odf, np.abs)

