"""
Functions used to calculate 2D and 3D resonances.
"""

from __future__ import division

import logging
logger = logging.getLogger('pygrfnn.resonances')

import math
from collections import defaultdict, namedtuple

import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from pygrfnn.utils import memoize, MemoizeMutable, cartesian


@memoize
def fareySequence(N, k=1):
    """
    Generate Farey sequence of order N, less than 1/k

    Args:
        N (``int``): Order of the sequence.
        k (``int``): ``1/k`` defined the point resonances are "attached" to.

    Referece:
        Add Rogelio Tomas paper
    """
    # assert type(N) == int, "Order (N) must be an integer"
    a, b = 0, 1
    c, d = 1, N
    seq = [(a,b)]
    while c/d <= 1/k:
        seq.append((c,d))
        tmp = int(math.floor((N+b)/d))
        a, b, c, d = c, d, tmp*c-a, tmp*d-b
    return seq


def fareyRatio(f, tol=0.001, N=10):
    """
    Calculate the farey approximation of a single number

    Args:
        f (``float``): number to approximate
        tol (``float``): allowed tolerance in the approximation
        N (``int``): maximum order of the approximation
    """

    def recursion(f, tol=0.001, a=0, b=1, c=1, d=1, depth=1):
        if f - a/b <= tol*f:
            return a, b
        if c/d - f <= tol*f:
            return c, d

        if N == depth:
            if f - a/b <= c/d - f:
                return a, b
            else:
                return c, d

        tmp_a, tmp_b = a+c, b+d

        if f == tmp_a/tmp_b:
            return tmp_a, tmp_b
        elif tmp_a/tmp_b > f:
            return recursion(f, tol, a, b, tmp_a, tmp_b, depth=depth+1)
        else:
            return recursion(f, tol, tmp_a, tmp_b, c, d, depth=depth+1)

    return recursion(f, tol=tol)


@memoize
def resonanceSequence(N, k):
    """
    Compute resonance sequence

    Arguments:
        - N (int): Order
        - k (int): denominator of the Farey frequency resonances are attached to
    """
    a, b = 0, 1
    c, d = k, N-k
    seq = [(a,b)]
    while d >= 0:
        seq.append((c,d))
        tmp = int(math.floor((N+b+a)/(d+c)))
        a, b, c, d = c, d, tmp*c-a, tmp*d-b
    return seq


def rationalApproximation(points, N, tol=1e-3, lowest_order_only=True):
    """
    Return rational approximations for a set of 2D points.

    For a set of points :math:`(x,y)` where :math:`0 < x,y \\leq1`, return all
    possible rational approximations :math:`(a,b,c) \\; a,b,c \\in \\mathbb{Z}`
    such that :math:`(x,y) \\approx (a/c, b/c)`.

    Arguments:
        points: 2D (L x 2) points to approximate
        N: max order

    Returns:
        ``dict``: Dictionary with ``points`` as *keys* and the corresponding
        ``set`` of tuples ``(a,b,c)`` as values.
    """
    L,_ = points.shape

    # since this solutions assumes a>0, a 'quick' hack to also obtain solutions
    # with a < 0 is to flip the dimensions of the points and explore those
    # solutions as well
    points = np.vstack((points, np.fliplr(points)))

    solutions = defaultdict(set)

    sequences = {1: set(fareySequence(1))}
    for n in range(2, N+1):
        sequences[n] = set(fareySequence(n)) - sequences[n-1]

    for h,k in fareySequence(N,1):
        if 0 in (h,k):
            continue
        # print h,k
        for x,y in resonanceSequence(N, k):

            # avoid 0-solutions
            if 0 in (x,y):
                continue

            norm = np.sqrt(x**2+y**2)

            n = np.array([ y/norm, x/norm]) * np.ones_like(points)
            n[points[:,0] < h/k, 0] *= -1  # points approaching from the left

            # nomenclature inspired in http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation
            ap = np.array([h/k, 0]) - points
            apn = np.zeros((1,L))
            d = np.zeros_like(points)

            apn = np.sum(n*ap, 1, keepdims=True)
            d = ap - apn*n

            ## DON'T RETURN IMMEDIATELY; THERE MIGHT BE OTHER SOLUTIONS OF THE SAME ORDER
            indices, = np.nonzero(np.sqrt(np.sum(d*d,1)) <= tol)
            for i in indices:
                # print "h/k:", h , "/", k
                # print "point:", points[i,:]
                if points[i,0] >= h/k:
                    if i<L:
                        # print "non-flipped >= h/k"
                        solutions[i].add((x,-y, h*x/k))
                        # print i, (x,-y, h*x/k)
                    elif x*(-y)<0:  # only consider solutions where (a,b) have different sign for the "flipped" points (the other solutions should have already been found for the non-flipped points)
                        # print "flipped >= h/k"
                        solutions[i-L].add((-y, x, h*x/k))
                        # print i-L, (-y, x, h*x/k)
                else:
                    if i<L:
                        # print "non-flipped < h/k"
                        solutions[i].add((x, y, h*x/k))
                        # print i, (x, y, h*x/k)
                    elif x*y>0:  # only consider solutions where (a,b) have different sign for the "flipped" points (the other solutions should have already been found for the non-flipped points)
                        # print "flipped < h/k"
                        solutions[i-L].add((y, x, h*x/k))
                        # print i-L, (y, x, h*x/k)

    if lowest_order_only:
        # removed = 0
        for k in solutions:
            # keep lowest order solutions only
            lowest_order = 2*N
            s = set([])
            for sol in solutions[k]:
                K = abs(sol[0])+abs(sol[1])+abs(sol[2])
                if K == lowest_order:
                    s.add(sol)
                elif K < lowest_order:
                    lowest_order = K
                    # if len(s) > 0:
                    #     print("point: ({},{}) -> removing {} for {}".format(points[k,0], points[k,1], s, sol))
                    #     removed += len(s)
                    s = set([sol])
            solutions[k] = s
        # print("Removed {} solutions".format(removed))

    return solutions


@MemoizeMutable
def threeFreqMonomials(fj, fi, allow_self_connect=True,
                       N=5, tol=1e-10, lowest_order_only=True):
    """
    Find the 3-tuples of frequencies such that

    .. math ::
        n_{ij1}f_{j1} + n_{ij2}f_{j2} \\approx d_{ij}f_{i}

    where :math:`n_{ij1}, n_{ij1}, d_{ij} \\in \\mathbb{Z}` and
    :math:`d_{ij} > 0`.

    Args:
        fj (:class:`numpy.ndarray`): frequency vector of the source (j in the paper)
        fi (:class:`numpy.ndarray`): frequency vector of the target (i in the paper)
        allow_self_connect (``bool``): if ``True``, :math:`n_{ij}` can be zero.
            Otherwise, non zero solutions are returned.
        N (``int``): max order of the monomials
        tol (``flaot``): tolerance in the approximation (see
            :func:`rationalApproximation`)
        lowest_order_only (``bool``): if ``True``, only monomials of the lowest
            order will be returned, even if there are higher order relationships
            that satisfy the equation.

    Returns:
        ``list``: List of :class:`python.namedtuples` of the from
        ``(indices, exponents)``. There is one ``namedtuple`` for each
        oscillator in ``destination`` GrFNN. Each tuple is formed by of two
        :class:`numpy.ndarray` with indices and exponents of a 3-freq monomial,
        each one of size Nx3, where N is the number of monomials associated to
        the corresponding oscillator in ``destination``.
    """
    from time import time
    st = time()

    fj = np.array([f for f in fj], dtype=np.float32)
    fi = np.array([f for f in fi], dtype=np.float32)
    Fj, Fi = len(fj), len(fi)

    cart_idx = cartesian((np.arange(Fj),
                          np.arange(Fj),
                          np.arange(Fi)))

    # we care only when y2 > y1
    cart_idx = cart_idx[cart_idx[:,1]>cart_idx[:,0]]

    if not allow_self_connect:
        cart_idx = cart_idx[(cart_idx[:,0] != cart_idx[:,2]) & (cart_idx[:,1] != cart_idx[:,2])]

    # actual frequency triplets
    cart = np.vstack((fj[cart_idx[:,0]], fj[cart_idx[:,1]], fi[cart_idx[:,2]])).T
    nr, _ = cart_idx.shape

    # sort in order to get a*x+b*y=c with 0<x,y<1
    sorted_idx = np.argsort(cart, axis=1)
    cart.sort()
    logger.info("a) Elapsed: {} secs".format(time() - st))
    all_points = np.zeros((nr, 2), dtype=np.float32)
    all_points[:,0] = cart[:,0] / cart[:,2]
    all_points[:,1] = cart[:,1] / cart[:,2]
    # del cart
    logger.info("b) Elapsed: {} secs".format(time() - st))

    redundancy_map = defaultdict(list)
    for i in xrange(all_points.shape[0]):
        redundancy_map[(all_points[i,0],all_points[i,1])].append(i)
    del all_points
    logger.info("c) Elapsed: {} secs".format(time() - st))

    points = np.array([[a,b] for a,b in redundancy_map])
    logger.info("d) Elapsed: {} secs".format(time() - st))

    exponents = rationalApproximation(points, N, tol=tol, lowest_order_only=lowest_order_only)
    logger.info("e) Elapsed: {} secs".format(time() - st))


    monomials = [defaultdict(list) for x in fi]
    M = namedtuple('Monomials', ['indices', 'exponents'])
    for k in exponents:
        x, y = points[k,0], points[k,1]
        all_points_idx = redundancy_map[(x,y)]
        sols = exponents[k]
        for a, b, c in sols:
            for idx in all_points_idx:
                j1, j2, i = (cart_idx[idx, 0], cart_idx[idx, 1], cart_idx[idx, 2])
                reordered = (sorted_idx[idx,0], sorted_idx[idx,1], sorted_idx[idx,2])
                if reordered == (0,1,2):
                    n1, n2, d = a, b, c
                elif reordered == (0,2,1):
                    # n1, d, n2 = -a, b, c
                    n1, n2, d = -a, c, b
                elif reordered == (2,0,1):
                    # d, n1, n2 = a, -b, c
                    n1, n2, d = -b, c, a
                else:
                    raise Exception("Unimplemented order!")
                if d < 0:
                    n1, n2, d = -n1, -n2, -d
                monomials[i]['j1'] += [j1]
                monomials[i]['j2'] += [j2+len(fj)]  # add offset for fast look-up at run time (will use a flattened array)
                monomials[i]['i']  += [i+2*len(fj)]   # add offset for fast look-up at run time (will use a flattened array)
                monomials[i]['n1'] += [n1]
                monomials[i]['n2'] += [n2]
                # monomials[i]['d']  += [d]
                monomials[i]['d']  += [d-1]

                # print i, (a,b,c), (n1, n2, d)

    logger.info("f) Elapsed: {} secs".format(time() - st))

    # make them 2d arrays for speed up
    for i, m in enumerate(monomials):
        I = np.array([m['j1'], m['j2'], m['i']], dtype=np.int).T
        E = np.array([m['n1'], m['n2'], m['d']], dtype=np.int).T
        # add trivial solution (1,0,1)
        if allow_self_connect:
            I = np.vstack((I,[i, i+Fj, i+2*Fj]))
            E = np.vstack((E,[1, 0, 0]))
        monomials[i] = M(indices=I, exponents=E)
    logger.info("g) Elapsed: {} secs".format(time() - st))

    return monomials
