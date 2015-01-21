#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Jorge Herrera
# @Date:   2015-01-19 21:07:50
# @Last Modified by:   jorgeh
# @Last Modified time: 2015-01-20 19:26:23

import numpy as np

from pygrfnn.oscillator import Zparam
from pygrfnn.grfnn import GrFNN
from pygrfnn.network import make_connections
from pygrfnn.network import Connection
from pygrfnn.network import Model


zp = Zparam(-1,  0,  0, 0, 0, 1)
layer_N50_narrow = GrFNN(zp, frequency_range=(0.5,2), num_oscs=50)
layer_N100_narrow = GrFNN(zp, frequency_range=(0.5,2), num_oscs=100)
layer_N50_wide = GrFNN(zp, frequency_range=(0.25,4), num_oscs=50)
layer_N100_wide = GrFNN(zp, frequency_range=(0.25,4), num_oscs=100)

def test_connection_same_size_layers():
    C = make_connections(layer_N50_narrow, layer_N50_wide, 1, 1)
    assert C.shape == (len(layer_N50_wide.f), len(layer_N50_narrow.f))

def test_connection_different_size_layers():
    C = make_connections(layer_N50_narrow, layer_N100_narrow, 1, 1)
    assert C.shape == (len(layer_N100_narrow.f), len(layer_N50_narrow.f))

def test_frequency_relationships():
    l1 = GrFNN(zp, frequency_range=(0.5,2), num_oscs=3)
    l2 = GrFNN(zp, frequency_range=(0.5,2), num_oscs=3)
    C = make_connections(l1, l2, 1, 1)
    model = Model()
    model.add_layer(l1)
    model.add_layer(l2)
    conn = model.connect_layers(l1, l2, C, '1freq')
    np.testing.assert_array_equal(conn.RF, np.array([[1., .5, .25],
                                                     [2., 1., .5],
                                                     [4., 2., 1]]))

    np.testing.assert_array_equal(conn.farey_num, np.array([[1., 1., 1.],
                                                            [2., 1., 1.],
                                                            [4., 2., 1]]))
    np.testing.assert_array_equal(conn.farey_den, np.array([[1., 2., 4.],
                                                            [1., 1., 2.],
                                                            [1., 1., 1]]))