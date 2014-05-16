import pygfnn.network as net
import pygfnn.oscillator as osc
import pygfnn.gfnn as gfnn

import numpy as np
import matplotlib.pyplot as plt


# load onset signal
odf = np.loadtxt('sampleOnsets/rumba.onset.txt')
fs_odf = odf[0]
odf = odf[1:]
t_odf = np.arange(0, odf.size)
t_odf = t_odf/fs_odf

# plt.plot(t_odf, odf)
# plt.show()



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