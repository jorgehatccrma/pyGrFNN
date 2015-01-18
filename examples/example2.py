# 0. Preliminares

import sys
sys.path.append('../')  # needed to run the examples from within the package folder

import numpy as np
import matplotlib.pyplot as plt

from pygrfnn.network import Model, make_connections, model_update_event
from pygrfnn.oscillator import Zparam
from pygrfnn.grfnn import GrFNN
from pygrfnn.vis import plot_connections
from pygrfnn.vis import tf_detail


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



model = Model(update_interval=0.005)
model.add_layer(layer1)  # layer one will receive the external stimulus (channel 0 by default)
model.add_layer(layer2, input_channel=None)  # layer 2 is a hidden layer (no externa input)

conn = model.connect_layers(layer1, layer2, C, '1freq')


# plt.plot(np.abs(C[len(layer2.f)/2,:]))
# plt.plot(np.angle(C[len(layer2.f)/2,:]))
# plot_connections(conn, title='Connection matrix (abs)')


plt.ion()


fig1 = plt.figure(2)
plt.clf()
ax1 = fig1.add_subplot(111)
ax1.grid(True)
line1, = ax1.semilogx(layer1.f, np.abs(layer1.z), 'k')
ax1.axis((np.min(layer1.f), np.max(layer1.f), 0, 1))

def update_callback1(sender, **kwargs):
    z = kwargs['z']
    line1.set_ydata(np.abs(z))
    ax1.set_title('t = {:0.2f}s'.format(kwargs['t']))
    fig1.canvas.draw()


fig2 = plt.figure(3)
plt.clf()
ax2 = fig2.add_subplot(111)
ax2.grid(True)
line2, = ax2.semilogx(layer2.f, np.abs(layer2.z), 'k')
ax2.axis((np.min(layer2.f), np.max(layer2.f), 0, 1))

def update_callback2(sender, **kwargs):
    z = kwargs['z']
    line2.set_ydata(np.abs(z))
    ax2.set_title('t = {:0.2f}s'.format(kwargs['t']))
    fig2.canvas.draw()



model_update_event.connect(update_callback1, sender=layer1)
model_update_event.connect(update_callback2, sender=layer2)




model.run(s, t, dt)