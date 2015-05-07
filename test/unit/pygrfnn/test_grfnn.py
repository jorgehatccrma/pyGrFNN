#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Jorge Herrera
# @Date:   2015-01-19 21:18:30
# @Last Modified by:   jorgeh
# @Last Modified time: 2015-05-07 00:07:19

import numpy as np

from pygrfnn.oscillator import Zparam
from pygrfnn.grfnn import GrFNN
from pygrfnn.network import make_connections
from pygrfnn.network import Connection
from pygrfnn.network import Model

import matplotlib.pyplot as plt

fs = 40.0
dt = 1/fs

linear_params = Zparam(-1,  0,  0, 0, 0, 1)  # Linear
critical_params = Zparam( 0, -1, -1, 0, 0, 1)  # Critical
crticial_detune_params = Zparam( 0, -1, -1, 1, 0, 1)  # Critical with detuning
limit_cycle_params = Zparam( 1, -1, -1, 0, 0, 1)  # Limit Cycle
double_limit_cycle_params = Zparam(-1, -3, -1, 0, 0, 1)  # Double Limit-cycle

linear_layer = GrFNN(linear_params,
                     frequency_range=(0.5,2),
                     num_oscs=200,
                     stimulus_conn_type='linear')

critical_layer = GrFNN(linear_params,
                       frequency_range=(0.5,2),
                       num_oscs=200,
                       stimulus_conn_type='linear')


def test_simple_network():
    t = np.arange(0,1,dt)
    f0 = 1.0
    s = 0.01*np.exp(1j*2*np.pi*f0*t)

    model = Model()
    model.add_layer(critical_layer)

    model.run(s, t, dt)

def test_two_layer_network():
    t = np.arange(0,1,dt)
    f0 = 1.0
    s = 0.01*np.exp(1j*2*np.pi*f0*t)

    C = make_connections(linear_layer,  # source layer
                         critical_layer,  # destination layer
                         1,  # connection strength multiplicative factor
                         0.0015
                         )


    model = Model()
    model.add_layer(linear_layer, input_channel=0)
    model.add_layer(critical_layer)

    # simmetric connections
    conn = model.connect_layers(linear_layer, critical_layer, C, 'allfreq')
    conn = model.connect_layers(critical_layer, linear_layer, C, 'allfreq')

    model.run(s, t, dt)
