#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: jorgeh
# @Date:   2015-01-25 11:07:19
# @Last Modified by:   jorgeh
# @Last Modified time: 2015-01-26 20:16:40

from __future__ import division

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

from pygrfnn.oscillator import Zparam
from pygrfnn.grfnn import GrFNN
from pygrfnn.network import Model, make_connections

np.set_printoptions(precision=4,
                    linewidth=20,
                    suppress=True  # print small results as 0
                    )

D = loadmat('test/testSignal')

sr = float(D['Fs'][0][0])
s = D['signal'][0]  # analytical signal (generated in matlab)

# s = s[:10]

# plt.plot(real(s))

num_oscs = 100

zp1 = Zparam(0.00001, 0, -2.0, 0, 0, 1)
zp2 = Zparam(-0.4, 1.75, -1.25, 0, 0, 1)

# n1 = GrFNN(zp1, name='L1', frequency_range=(1., 4.), num_oscs=num_oscs, z0=0.5+0.5j, stimulus_conn_type='active')
# n2 = GrFNN(zp2, name='L2', frequency_range=(1., 4.), num_oscs=num_oscs, z0=0.5+0.5j)
n1 = GrFNN(zp1, name='L1', frequency_range=(1., 4.), num_oscs=num_oscs, stimulus_conn_type='active')
n2 = GrFNN(zp2, name='L2', frequency_range=(1., 4.), num_oscs=num_oscs)


M = Model()
M.add_layer(n1, input_channel=0)
M.add_layer(n2)


# Connections
modes = [1/3, 1/2, 1/1, 2/1, 3/1]
amps  = [1,   1,   1,   1,   1  ]

C11 = make_connections(n1, n1, 1.0,  1.05, modes=modes, mode_amps=amps)
C12 = make_connections(n1, n2, 1.0,  1.05, modes=modes, mode_amps=amps)
C22 = make_connections(n2, n2, 1.0,  1.05, modes=modes, mode_amps=amps)
C21 = make_connections(n2, n1, 1.0,  1.05, modes=modes, mode_amps=amps)

c11 = M.connect_layers(n1, n1, C11, '2freq', weight=.10)
c12 = M.connect_layers(n1, n2, C12, '2freq', weight=.40)
c22 = M.connect_layers(n2, n2, C22, '2freq', weight=.10)
c21 = M.connect_layers(n2, n1, C21, '2freq', weight=.05)


M.run(s, np.arange(len(s), dtype=float)/sr, 1./sr)

plt.ion()
plt.clf()
plt.plot(n1.f, np.abs(n1.TF),'b')
plt.plot(n2.f, np.abs(n2.TF),'g')
