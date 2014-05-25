"""
Script to study the dynamics of a single oscillator
"""

import pygfnn
import numpy as np
import matplotlib.pyplot as plt



TWO_PI = pygfnn.TWO_PI



# oscillator parameters
zparams = pygfnn.Zparam(alpha=-0.3,
                        beta1=2.0,
                        beta2=-2.0,
                        epsilon=0.5)

# a single neural oscillator
osc = pygfnn.GFNN(zparams,
                  fc=2.0,
                  octaves_per_side=0.0)

model = pygfnn.Model()
model.add_layer(osc)


dur = 50.0
fs = 100.0
dt = 1.0/fs
t = np.arange(0, dur, dt)
f = 2.0
s = 0.5*np.sin(t*TWO_PI*f)

mask = np.ones(t.shape)
mask[t>1.5] = 0
mask[t>6] = 1
mask[t>30] = 0


s = s*mask

model.run(s, t, dt)
z = osc.TF


plt.plot(t, s)
plt.hold(True)
plt.plot(t, np.real(z).T)
plt.hold(False)

plt.show()