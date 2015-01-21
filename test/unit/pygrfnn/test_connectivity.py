#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Jorge Herrera
# @Date:   2015-01-19 21:07:50
# @Last Modified by:   Jorge Herrera
# @Last Modified time: 2015-01-19 21:35:53

from nose import with_setup

from pygrfnn.oscillator import Zparam
from pygrfnn.grfnn import GrFNN
from pygrfnn.network import make_connections
from pygrfnn.network import Connection


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
