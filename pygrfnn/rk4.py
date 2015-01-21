"""
Theano implementation of RK4

Source: https://github.com/andyr0id/theano-rk4
"""

import numpy as np
import theano.tensor as T
from theano import function
from theano import Param
from theano import pp
import matplotlib.pyplot as plt


def rk4_step_exp(t0, y0, h, dydt_exp, *args):
    half_h = h/2

    k1 = h * dydt_exp(t0, y0, *args)

    t2 = t0 + half_h
    y2 = y0 + (k1/2)
    k2 = h * dydt_exp(t2, y2, *args)

    y3 = y0 + (k2/2)
    k3 = h * dydt_exp(t2, y3, *args)

    t4 = t0 + h
    y4 = y0 + k3
    k4 = h * dydt_exp(t4, y4, *args)

    yi = y0 + (k1 + 2*k2 + 2*k3 + k4)/6
    return yi

if __name__ == '__main__':

    def vdpo_exp(t,y,mu):
        #vdpo in theano
        y1 = y[1]
        y2 = mu * (1-y[0]**2) * y[1] - y[0]
        return [y1, y2]

    mu = T.dscalar('mu') # vdpo param
    t0 = T.dscalar('t0') # time at integration step
    y0 = T.dvector('y0') # state at integration step
    h = T.dscalar('h') # fixed time step
    rk4_step = rk4_step_exp(t0, y0, h, vdpo_exp, mu)

    # make theano function, on_unused_input is required as the vdpo does not use time
    vdpo_rk4_step_fn = function([t0, y0, h, mu], rk4_step, on_unused_input='ignore')

    # integrate a certain amount of time
    y = [0.1,0.]
    time = 50
    step = 0.001
    n_steps = int(time / step)
    t = 0
    T = np.zeros(n_steps)
    Y = np.zeros((n_steps,len(y)))
    for i in xrange(n_steps):
        t = i * step
        y = vdpo_rk4_step_fn(t,y,step,10)
        T[i] = t
        Y[i, :] = y

    # plot the result
    fig1 = plt.figure()
    plt.plot(T, Y[:,0])
    plt.show()