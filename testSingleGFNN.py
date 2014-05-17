import pygfnn.network as net
import pygfnn.oscillator as osc
import pygfnn.gfnn as gfnn
from pygfnn.utils import nextpow2

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


plot_onset_signal = False
plot_internal_conns = False
plot_tf_output = True


# load onset signal
odf = np.loadtxt('sampleOnsets/rumba.onset.txt')
fs_odf = odf[0]
odf = odf[1:]
t_odf = np.arange(0, odf.size)
t_odf = t_odf/fs_odf


if plot_onset_signal:
    plt.plot(t_odf, odf)
    plt.show()



# create single gfnn model

params = osc.Zparam(0, -0.25, -0.5, 0, 0, 1)

internal_strength = 0.6
internal_stdev = 0.5

center_freq = 2.0
half_range =  2
oscs_per_octave = 64


layer = gfnn.GFNN(params,
                  fc=center_freq,
                  octaves_per_side=half_range,
                  oscs_per_octave=oscs_per_octave,
                  internal_strength=internal_strength,
                  internal_stdev=internal_stdev)

net = net.Model()
net.add_layer(layer)


# plot internal conns
conn = layer.internal_conns[np.round(layer.f.size/2.0)]



if plot_internal_conns:
    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)

    ax1.semilogx(layer.f, np.abs(conn))
    ax1b = ax1.twinx()
    ax1b.semilogx(layer.f, np.angle(conn), color='r')
    ax1.axis('tight')


    ax2.semilogx(layer.f, np.real(conn))
    ax2b = ax2.twinx()
    ax2b.semilogx(layer.f, np.imag(conn), color='r')
    ax2.axis('tight')

    plt.show()



# run the model

# # temporal hack
# odf = np.insert(odf, 0, 0)
# odf = odf[:-1]

net.process_signal(odf*0.5, t_odf, 1.0/fs_odf)
TF = layer.TF
f = layer.f
T = 1.0/f


if plot_tf_output:
    fig, axTF = plt.subplots(1)
    im = axTF.pcolormesh(t_odf, T, np.abs(TF), cmap='gray_r')
    axTF.set_yscale('log')
    # axTF.set_yticks(np.min(T)*(2**np.arange(2*oscs_per_octave+1)))
    axTF.set_yticks(2**np.arange(np.log2(nextpow2(np.min(T))), 1+np.log2(nextpow2(np.max(T)))))
    axTF.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    # axTF.set_xticklabels([])
    axTF.axis('tight')

    plt.show()
