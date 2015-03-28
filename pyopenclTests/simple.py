#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Learning PyOpenCL
"""

import numpy as np
import pyopencl as cl


fs = 48000.0
f0 = 440.0
dur = 5.0   # duration in seconds
n = np.arange(dur*fs)

x_cpu = np.sin(2.*np.pi*f0*n/fs).astype(float32)


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
x_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_cpu)

prg = cl.Program(ctx, """
__kernel void sum(__global const float *a_g, __global const float *b_g, __global float *res_g) {
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
""").build()

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)

# Check on CPU with Numpy:
print(res_np - (a_np + b_np))
print(np.linalg.norm(res_np - (a_np + b_np)))