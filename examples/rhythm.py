"""
Rhythm processing model
"""

%% Networks
n1 = networkMake(1, 'hopf', alpha1, beta11, beta12, 0,  0,  neps1, ...
                    'log', .375, 12, 321, ...
                    'display', 2, 'save', 1, 'channel', 1);
n2 = networkMake(2, 'hopf', alpha2, beta21, beta22,  0, 0, neps2, ...
                    'log', .375, 12, 321, ...
                    'display', 2, 'save', 1, 'channel', 1);
n1.w = 3.0;

%% Connections
modes =      [1/3 1/2 1/1 2/1 3/1];
amps  =      [1   1   1   1   1 ];
sds   = 1.05*[1   1   1   1   1  ];

C1 = connectMake(n1, n1, 'gaus',  1, 1.05, 0, 1, 'modes', modes, 'amps', amps, 'sds', sds);
C0 = connectMake(n1, n1, 'gaus',  1, 1.05, 0, 0, 'modes', modes, 'amps', amps, 'sds', sds);

%% Connections
% Internal
n1 = connectAdd(n1, n1, C0, 'weight', .10, 'type', '2freq');

% Afferent
n2 = connectAdd(n1, n2, C1, 'weight', .40, 'type', '2freq');

% Internal
n2 = connectAdd(n2, n2, C0, 'weight', .10, 'type', '2freq');

% Efferent
n1 = connectAdd(n2, n1, C1, 'weight', .05, 'type', '2freq');

%% Model
M = modelMake(@zdot, @cdot, s, n1, n2);
figure(2); imagesc(M.n{1}.con{1}.C); colormap(flipud(hot)); colorbar;


# # 0. Preliminares

# import sys
# sys.path.append('../')  # needed to run the examples from within the package folder

# import numpy as np
# import matplotlib.pyplot as plt

# from pygrfnn.network import Model, make_connections
# from pygrfnn.oscillator import Zparam
# from pygrfnn.grfnn import GrFNN
# from pygrfnn.vis import plot_connections
# from pygrfnn.vis import tf_detail
# from pygrfnn.vis import GrFNN_RT_plot

# # 1. Create Stimulus: Complex sinusoid

# sr = 4000.0  # sample rate
# dt = 1.0/sr
# t = np.arange(0, 1, dt)
# fc = 100.0  # frequency
# A = 0.025  # amplitude
# s = A * np.exp(1j * 2 * np.pi * fc * t)

# # ramp signal linearly up/down
# ramp_dur = 0.01  # in secs
# ramp = np.arange(0, 1, dt / ramp_dur)
# env = np.ones(s.shape, dtype=float)
# env[0:len(ramp)] = ramp
# env[-len(ramp):] = ramp[::-1]
# # apply envelope
# s = s * env

# # plot stimulus
# plt.ion()
# plt.plot(t, np.real(s))
# plt.plot(t, np.imag(s))
# plt.title('Stimulus')


# # Explore different parameter sets

# params1 = Zparam(0.01,-1.,-10., 0., 0., 1.)  # Linear
# params2 = Zparam( -1., 4., -3., 0., 0., 1.)  # Critical


# # Make the model
# layer1 = GrFNN(params1,
#                frequency_range=(50,200),
#                num_oscs=200,
#                stimulus_conn_type='active')

# layer2 = GrFNN(params2,
#                frequency_range=(50,200),
#                num_oscs=200)

# # C = make_connections(layer1,  # source layer
# #                      layer2,  # destination layer
# #                      1,  # connection strength multiplicative factor
# #                      0.028712718  # std dev(eye-balled to closely match that of GrFNN =-Toolbox-1.0 example)
# #                      )
# C = make_connections(layer1,  # source layer
#                      layer2,  # destination layer
#                      1,  # connection strength multiplicative factor
#                      0.0015
#                      )



# model = Model()
# model.add_layer(layer1)  # layer one will receive the external stimulus (channel 0 by default)
# model.add_layer(layer2, input_channel=None)  # layer 2 is a hidden layer (no externa input)

# conn = model.connect_layers(layer1, layer2, C, '1freq')


# # plt.plot(np.abs(C[len(layer2.f)/2,:]))
# # plt.plot(np.angle(C[len(layer2.f)/2,:]))
# # plot_connections(conn, title='Connection matrix (abs)')


# plt.ion()

# GrFNN_RT_plot(layer1, update_interval=0.005, title='First Layer')
# GrFNN_RT_plot(layer2, update_interval=0.005, title='Second Layer')

# model.run(s, t, dt)