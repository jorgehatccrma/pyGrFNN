# 0. Preliminares

import sys
sys.path.append('../')  # needed to run the examples from within the package folder

import numpy as np
import matplotlib.pyplot as plt

from pygrfnn.network import Model, make_connections
from pygrfnn.oscillator import Zparam
from pygrfnn.grfnn import GrFNN
from pygrfnn.vis import plot_connections
from pygrfnn.vis import tf_detail
from pygrfnn.vis import GrFNN_RT_plot

# 1. Create Stimulus: Complex sinusoid

sr = 4000.0  # sample rate
dt = 1.0/sr
t = np.arange(0, 1, dt)
fc = 100.0  # frequency
A = 0.025  # amplitude
s = A * np.exp(1j * 2 * np.pi * fc * t)

# ramp signal linearly up/down
ramp_dur = 0.01  # in secs
ramp = np.arange(0, 1, dt / ramp_dur)
env = np.ones(s.shape, dtype=float)
env[0:len(ramp)] = ramp
env[-len(ramp):] = ramp[::-1]
# apply envelope
s = s * env

# plot stimulus
plt.ion()
plt.plot(t, np.real(s))
plt.plot(t, np.imag(s))
plt.title('Stimulus')


# Explore different parameter sets

params1 = Zparam(0.01,-1.,-10., 0., 0., 1.)  # Linear
params2 = Zparam( -1., 4., -3., 0., 0., 1.)  # Critical


# Make the model
layer1 = GrFNN(params1,
               frequency_range=(50,200),
               num_oscs=200,
               stimulus_conn_type='active')

layer2 = GrFNN(params2,
               frequency_range=(50,200),
               num_oscs=200)

# C = make_connections(layer1,  # source layer
#                      layer2,  # destination layer
#                      1,  # connection strength multiplicative factor
#                      0.028712718  # std dev(eye-balled to closely match that of GrFNN =-Toolbox-1.0 example)
#                      )
C = make_connections(layer1,  # source layer
                     layer2,  # destination layer
                     1,  # connection strength multiplicative factor
                     0.0015
                     )



model = Model()
model.add_layer(layer1, input_channel=0)  # layer one will receive the external stimulus
model.add_layer(layer2)  # layer 2 is a hidden layer (no external input)

conn = model.connect_layers(layer1, layer2, C, '1freq')


# plt.plot(np.abs(C[len(layer2.f)/2,:]))
# plt.plot(np.angle(C[len(layer2.f)/2,:]))
# plot_connections(conn, title='Connection matrix (abs)')


plt.ion()

GrFNN_RT_plot(layer1, update_interval=0.005, title='First Layer')
GrFNN_RT_plot(layer2, update_interval=0.005, title='Second Layer')

model.run(s, t, dt)