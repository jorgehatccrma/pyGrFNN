"""
Script to test agogic accentuation
"""

from __future__ import division

import numpy as np
from time import time
from scipy.signal import hilbert
import matplotlib.pyplot as plt

from daspy import Signal
from daspy.audioio import play_signal
from daspy.processing import onset_detection_signal
from pyrhythm import Pattern

from rhythm import rhythm_model

import matplotlib.pyplot as plt
from pygrfnn.vis import plot_connections
from pygrfnn.vis import tf_detail
from pygrfnn.vis import GrFNN_RT_plot


N = 4       # numerator
D = 4       # denominator
d = D/N     # even event period
# c = 0.025     # percentage deviation
c = 0.0     # percentage deviation

d_star = (1+c)*d  # duration of (after) down beat
d_prime = (D - d_star) / (N-1)  # duration of other events

agogic_pattern = np.array([0, d_star]+[d_prime]*(N-1))

print agogic_pattern
print np.cumsum(agogic_pattern)

p = Pattern(agogic_pattern, numerator=N, denominator=D)

sr = 22050
x, _ = p.as_signal(120, reps=10, sr=sr)
# t = np.arange(len(x))/sr
# plt.plot(t, x)
# plt.show()


x = Signal(x, sr=sr)
x = 0.99*x.normalize()


play_pattern = False
if play_pattern:
    play_signal(x)

s = onset_detection_signal(x)
rms = np.sqrt(np.sum(s**2)/len(s))
s *= 0.06/rms
s = Signal(hilbert(s), sr=s.sr)
t = s.time_vector()
dt = 1/s.sr


model = rhythm_model()
layer1, layer2 = model.layers()

# Simulation
RT_display = True
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

