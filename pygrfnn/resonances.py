
from __future__ import division

import logging
logger = logging.getLogger('pygrfnn.resonances')

import math
from collections import defaultdict, namedtuple

import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from pygrfnn.utils import memoize, MemoizeMutable, cartesian

# def findResonance(f1, f2, f0):
#     """
#     find a*f1 + b*f2 = c * f0 (approximately equal)
#     """
#     a, b, c = fareySequence(f1/f0, f2/f0)


# @memoize
# def fareySequence(Qx, Qy, max_order=16):
#     pass


@memoize
def fareySequence(N, k=1):
    """
    Generate Farey sequence of order N, less than 1/k
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


def fareyRatio(f, tol=0.001, max_order=10):

    def recursion(f, tol=0.001, a=0, b=1, c=1, d=1, depth=1):
        if f - a/b <= tol*f:
            return a, b
        if c/d - f <= tol*f:
            return c, d

        if max_order == depth:
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
        - k (int): denominator of the farey frequency resonances are attached to
    """
    a, b = 0, 1
    c, d = k, N-k
    seq = [(a,b)]
    while d >= 0:
        seq.append((c,d))
        tmp = int(math.floor((N+b+a)/(d+c)))
        a, b, c, d = c, d, tmp*c-a, tmp*d-b
    return seq


def plotResonanceDiagram(N, exclude_inf=True):
    import matplotlib.pyplot as plt

    ALPHA = 0.2

    plt.figure()
    ticks = set([])
    for h, k in fareySequence(N, 1):
        ticks.add((h,k))
        for a, b in resonanceSequence(N, k):
            if b == 0:
                if not exclude_inf:
                    plt.plot([h/k, h/k], [0, 1], 'b:', alpha=2*ALPHA)
                    plt.plot([0, 1], [h/k, h/k], 'b:', alpha=2*ALPHA)
                continue
            m = a/b
            cp, cm = m*h/k, -m*h/k
            x = np.array([0, h/k, 1])
            y = np.array([cp, 0, cm+m])
            plt.plot(  x,   y, 'b', alpha=ALPHA)
            plt.plot(  y,   x, 'b', alpha=ALPHA)
            plt.plot(  x, 1-y, 'b', alpha=ALPHA)
            plt.plot(1-y,   x, 'b', alpha=ALPHA)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([h/k for h,k in ticks], [r"$\frac{{{:d}}}{{{:d}}}$".format(h,k) for h,k in ticks])
    plt.yticks([h/k for h,k in ticks], [r"$\frac{{{:d}}}{{{:d}}}$".format(h,k) for h,k in ticks])
    # plt.xticks([h/k for h,k in ticks], [r"${:d}/{:d}$".format(h,k) for h,k in ticks])
    # plt.yticks([h/k for h,k in ticks], [r"${:d}/{:d}$".format(h,k) for h,k in ticks])
    plt.title("N = {:d}".format(N))



# def h_debug(Qx, Qy, M, tol=1e-3, debug=True):
#     """
#     Arguments:
#         Qx, Qy: 2D point to approximate
#         M: max order
#     """

#     bounds = {}

#     if debug:
#         import matplotlib.pyplot as plt
#         plt.figure();
#         plt.plot(Qx, Qy, 'ro');
#         current_line, = plt.plot([0, 1], [0, 0], 'k', alpha=0.4)
#         current_left, = plt.plot([0, 1], [0, 0], 'g:', alpha=0.4)
#         current_right, = plt.plot([0, 1], [0, 0], 'b:', alpha=0.4)
#         current_d, = plt.plot([0, 0], [0, 0], 'm', alpha=0.6)
#         pt_hk, = plt.plot(0, 0, 'mo')
#         lines = {}
#         plt.xlim(0,1)
#         plt.ylim(0,1)

#     N = 1
#     solutions = []
#     best_d = [np.array([1,1]), None]

#     if debug:
#         print("Searching for (Qx,Qy)=({},{})".format(Qx,Qy))

#     while N <= M:
#         # if debug:
#         #     print("N={}".format(N))
#         for h,k in fareySequence(N,1):
#             if 0 in (h,k):
#             # if k == 0:
#                 # if debug:
#                 #     print("skipping h/k={}/{}".format(h,k))
#                 continue
#             if debug:
#                 # print("h/k={}/{}".format(h,k))
#                 pt_hk.set_xdata(h/k);

#             if (h,k) not in bounds:
#                 bounds[(h,k)] = {'left':np.array([-1,0]), 'right':np.array([1,0])}
#                 if debug:
#                     lines[(h,k)] = {}
#                     lines[(h,k)]['left_line'], = plt.plot(None, None, 'g', alpha=0.2)
#                     lines[(h,k)]['right_line'], = plt.plot(None, None, 'b', alpha=0.2)

#             # TODO: use binary search for speed up? Maybe not worth ...

#             left = bounds[(h,k)]['left']
#             right = bounds[(h,k)]['right']

#             for x,y in resonanceSequence(N, k):

#                 if debug:
#                     current_left.set_xdata([h/k, h/k+left[0]])
#                     current_left.set_ydata([0, left[1]])
#                     current_right.set_xdata([h/k, h/k+right[0]])
#                     current_right.set_ydata([0, right[1]])

#                 # avoid 0-solutions
#                 if 0 in (x,y):
#                     continue

#                 norm = np.sqrt(x**2+y**2)

#                 if h/k >= Qx:        # approach from the right
#                     a, b = x, y
#                 else:                # approach from the left
#                     a, b = x,-y

#                 n = np.array([ -b/norm, a/norm])


#                 if n.dot(np.array([-left[1], left[0]])) < 0 and \
#                    n.dot(np.array([-right[1], right[0]])) > 0:
#                     if debug:
#                         current_line.set_xdata([h/k, h/k+2*n[0]])
#                         current_line.set_ydata([0, 2*n[1]])

#                         plt.plot([h/k, h/k+2*n[0]], [0, 2*n[1]], 'k', alpha=0.3)

#                     # nomenclature inspired in http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation
#                     av = np.array([h/k-Qx, -Qy])
#                     tmp = n.dot(av)
#                     d = av-tmp*n

#                     current_d.set_xdata([Qx, Qx+d[0]])
#                     current_d.set_ydata([Qy, Qy+d[1]])

#                     if d.dot(d) < best_d[0].dot(best_d[0]):
#                         best_d[0] = d
#                         best_d[1] = ((a, b, h*a/k))

#                     if np.sqrt(d.dot(d)) <= tol:
#                         ## DON'T RETURN IMMEDIATELY; THERE MIGHT BE OTHER SOLUTIONS OF THE SAME ORDER
#                         # return (a, b, h*a/k)
#                         solutions.append((a, b, h*a/k))
#                     # the following implicitly assumes we are in the positive quadrant
#                     elif d[0] < 0:
#                         left = bounds[(h,k)]['left'] = n

#                         if debug:
#                             lines[(h,k)]['left_line'].set_xdata(current_line.get_xdata())
#                             lines[(h,k)]['left_line'].set_ydata(current_line.get_ydata())
#                     else:
#                         right = bounds[(h,k)]['right'] = n
#                         if debug:
#                             lines[(h,k)]['right_line'].set_xdata(current_line.get_xdata())
#                             lines[(h,k)]['right_line'].set_ydata(current_line.get_ydata())
#                     if debug:
#                         plt.show()
#                         raw_input("Press Enter to continue...")

#         N += 1
#     if len(solutions) > 0:
#         if debug:
#             for a,b,c in solutions:
#                 plt.plot([c/a, c/a-b], [0, a], 'r')
#         print N, np.sqrt(best_d[0].dot(best_d[0])), best_d[1]
#         return solutions
#     # print np.sqrt(best_d[0].dot(best_d[0])), best_d[1]
#     # raise Exception("Max order reached")
#     return solutions


# def h(Qx, Qy, M, tol=1e-3):
#     """
#     Arguments:
#         Qx, Qy: 2D point to approximate
#         M: max order
#     """

#     bounds = {}

#     N = 1
#     solutions = []
#     best_d = [np.array([1,1]), None]

#     while N <= M:
#         for h,k in fareySequence(N,1):
#             if 0 in (h,k):
#                 continue

#             if (h,k) not in bounds:
#                 bounds[(h,k)] = {'left':np.array([-1,0]), 'right':np.array([1,0])}

#             left = bounds[(h,k)]['left']
#             right = bounds[(h,k)]['right']

#             for x,y in resonanceSequence(N, k):

#                 # avoid 0-solutions
#                 if 0 in (x,y):
#                     continue

#                 norm = np.sqrt(x**2+y**2)

#                 if h/k >= Qx:        # approach from the right
#                     a, b = x, y
#                 else:                # approach from the left
#                     a, b = x,-y

#                 n = np.array([ -b/norm, a/norm])


#                 if n.dot(np.array([-left[1], left[0]])) < 0 and \
#                    n.dot(np.array([-right[1], right[0]])) > 0:

#                     # nomenclature inspired in http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation
#                     av = np.array([h/k-Qx, -Qy])
#                     tmp = n.dot(av)
#                     d = av-tmp*n
#                     if d.dot(d) < best_d[0].dot(best_d[0]):
#                         best_d[0] = d
#                         best_d[1] = ((a, b, h*a/k))

#                     if np.sqrt(d.dot(d)) <= tol:
#                         ## DON'T RETURN IMMEDIATELY; THERE MIGHT BE OTHER SOLUTIONS OF THE SAME ORDER
#                         # return (a, b, h*a/k)
#                         solutions.append((a, b, h*a/k))
#                     # the following implicitly assumes we are in the positive quadrant
#                     elif d[0] < 0:
#                         left = bounds[(h,k)]['left'] = n
#                     else:
#                         right = bounds[(h,k)]['right'] = n

#         if len(solutions) > 0:
#             # print np.sqrt(best_d[0].dot(best_d[0])), best_d[1]
#             return solutions
#         N += 1
#     # print np.sqrt(best_d[0].dot(best_d[0])), best_d[1]
#     # raise Exception("Max order reached")
#     return solutions



def rationalApproximation(points, N, tol=1e-3, lowest_order_only=True):
    """
    Arguments:
        points: 2D (L x 2) points to approximate
        N: max order
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
def monomialsForVectors(fj, fi, allow_self_connect=True, N=5, tol=1e-10, lowest_order_only=True):
    """
    Arguments:
        fj (np.array_like): frequency vector of the source (j in the paper)
        fi (np.array_like): frequency vector of the target (i in the paper)
        N: max order
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



# if __name__ == '__main__':
#     from time import time
#     import pprint
#     pp = pprint.PrettyPrinter(indent=4)

#     max_order = 16
#     tol = 1e-3

#     points = np.random.rand(100,2)
#     # points = np.array([[ 0.20127775,  0.39311277]])

#     num_points = points.shape[0]

#     tmp0 = defaultdict(set)
#     st = time()
#     for i in range(num_points):
#         tmp = h(points[i,0], points[i,1], max_order, tol=tol)
#         if len(tmp) > 0:
#             tmp0[i] = set(tmp)
#     print("h took {} secs (found {:.2f}%)".format(time()-st, 100*len(tmp0)/num_points))

#     # st = time()
#     # tmp1 = findMonomials(points, max_order, tol=tol)
#     # print("findMonomials took {} secs (found {:.2f}%)".format(time()-st, 100*len(tmp1)/num_points))

#     st = time()
#     tmp2 = rationalApproximation(points, max_order, tol=tol)
#     print("rationalApproximation took {} secs (found {:.2f}%)".format(time()-st, 100*len(tmp2)/num_points))

#     # pp.pprint(points)
#     # pp.pprint(tmp0)
#     # pp.pprint(tmp1)
#     # pp.pprint(tmp2)

#     # for k in tmp1:
#     #     px, py = points[k,0], points[k,1]
#     #     for a,b,c in tmp1[k]:
#     #         print "{:.5f} = {} \t {:.5f} ; {:.5f} ".format(a*px+b*py, c, (a*px+b*py)/c, a*px+b*py-c)

#     clean = True
#     for k in tmp0:
#         if len(tmp0[k]-tmp2[k]) != 0:
#             clean = False
#             print("Error in index {}".format(k))
#             print(tmp0[k])
#             print(tmp2[k])

#     if clean:
#         print("SUCCESS!!!")
