#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: jorgeh
# @Date:   2015-01-25 11:07:19
# @Last Modified by:   jorgeh
# @Last Modified time: 2015-01-25 18:32:52

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

s = s[:9]

# plt.plot(real(s))

zp1 = Zparam(0.1, -1.0, -10., 0, 0, 1)
zp2 = Zparam(-0.1, 2.0, -3., 0, 0, 1)

n1 = GrFNN(zp1, name='L1', frequency_range=(1., 4.), num_oscs=5, z0=0.5+0.5j, stimulus_conn_type='active')
n2 = GrFNN(zp2, name='L2', frequency_range=(1., 4.), num_oscs=5, z0=0.5+0.5j)

# C = np.eye(len(n1.f), len(n2.f))
C = make_connections(n1, n2, 1., 1.005)

M = Model()
M.add_layer(n1, input_channel=0)
M.add_layer(n2)

M.connect_layers(n1, n2, C, connection_type='2freq',
                 self_connect=False  # to match connectAdd.m
                 )

M.run(s, np.arange(len(s), dtype=float)/sr, 1./sr)