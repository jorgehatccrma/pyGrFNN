
from __future__ import division
import math
from collections import defaultdict

import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from pygrfnn.utils import memoize, cartesian

# def findResonance(f1, f2, f0):
#     """
#     find a*f1 + b*f2 = c * f0 (approximately equal)
#     """
#     a, b, c = fareySequence(f1/f0, f2/f0)


# @memoize
# def fareySequence(Qx, Qy, max_order=16):
#     pass


# @memoize
def fareySequence(N, k=1):
    """
    Generate Farey sequence of order N, less than 1/k
    """
    assert type(N) == int, "Order (N) must be an integer"
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


def resonanceSequence(N, k):
    """
    Compute resonance sequence

    Arguments:
        - N (int): Order
        - k (int): denominator
    """

    pq = fareySequence(N, k)
    return [(k*p, q - k*p) for p, q in pq]


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



def h_debug(Qx, Qy, M, tol=1e-3, debug=True):
    """
    Arguments:
        Qx, Qy: 2D point to approximate
        M: max order
    """

    bounds = {}

    if debug:
        import matplotlib.pyplot as plt
        plt.figure();
        plt.plot(Qx, Qy, 'ro');
        current_line, = plt.plot([0, 1], [0, 0], 'k', alpha=0.4)
        current_left, = plt.plot([0, 1], [0, 0], 'g:', alpha=0.4)
        current_right, = plt.plot([0, 1], [0, 0], 'b:', alpha=0.4)
        current_d, = plt.plot([0, 0], [0, 0], 'm', alpha=0.6)
        pt_hk, = plt.plot(0, 0, 'mo')
        lines = {}
        plt.xlim(0,1)
        plt.ylim(0,1)

    N = 1
    solutions = []
    best_d = [np.array([1,1]), None]

    if debug:
        print("Searching for (Qx,Qy)=({},{})".format(Qx,Qy))

    while N <= M:
        # if debug:
        #     print("N={}".format(N))
        for h,k in fareySequence(N,1):
            if 0 in (h,k):
            # if k == 0:
                # if debug:
                #     print("skipping h/k={}/{}".format(h,k))
                continue
            if debug:
                # print("h/k={}/{}".format(h,k))
                pt_hk.set_xdata(h/k);

            if (h,k) not in bounds:
                bounds[(h,k)] = {'left':np.array([-1,0]), 'right':np.array([1,0])}
                if debug:
                    lines[(h,k)] = {}
                    lines[(h,k)]['left_line'], = plt.plot(None, None, 'g', alpha=0.2)
                    lines[(h,k)]['right_line'], = plt.plot(None, None, 'b', alpha=0.2)

            # TODO: use binary search for speed up? Maybe not worth ...

            left = bounds[(h,k)]['left']
            right = bounds[(h,k)]['right']

            for x,y in resonanceSequence(N, k):

                if debug:
                    current_left.set_xdata([h/k, h/k+left[0]])
                    current_left.set_ydata([0, left[1]])
                    current_right.set_xdata([h/k, h/k+right[0]])
                    current_right.set_ydata([0, right[1]])

                # avoid 0-solutions
                if 0 in (x,y):
                    continue

                norm = np.sqrt(x**2+y**2)

                if h/k >= Qx:        # approach from the right
                    a, b = x, y
                else:                # approach from the left
                    a, b = x,-y

                n = np.array([ -b/norm, a/norm])


                if n.dot(np.array([-left[1], left[0]])) < 0 and \
                   n.dot(np.array([-right[1], right[0]])) > 0:
                    if debug:
                        current_line.set_xdata([h/k, h/k+2*n[0]])
                        current_line.set_ydata([0, 2*n[1]])

                        plt.plot([h/k, h/k+2*n[0]], [0, 2*n[1]], 'k', alpha=0.3)

                    # nomenclature inspired in http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation
                    av = np.array([h/k-Qx, -Qy])
                    tmp = n.dot(av)
                    d = av-tmp*n

                    current_d.set_xdata([Qx, Qx+d[0]])
                    current_d.set_ydata([Qy, Qy+d[1]])

                    if d.dot(d) < best_d[0].dot(best_d[0]):
                        best_d[0] = d
                        best_d[1] = ((a, b, h*a/k))

                    if np.sqrt(d.dot(d)) <= tol:
                        ## DON'T RETURN IMMEDIATELY; THERE MIGHT BE OTHER SOLUTIONS OF THE SAME ORDER
                        # return (a, b, h*a/k)
                        solutions.append((a, b, h*a/k))
                    # the following implicitly assumes we are in the positive quadrant
                    elif d[0] < 0:
                        left = bounds[(h,k)]['left'] = n

                        if debug:
                            lines[(h,k)]['left_line'].set_xdata(current_line.get_xdata())
                            lines[(h,k)]['left_line'].set_ydata(current_line.get_ydata())
                    else:
                        right = bounds[(h,k)]['right'] = n
                        if debug:
                            lines[(h,k)]['right_line'].set_xdata(current_line.get_xdata())
                            lines[(h,k)]['right_line'].set_ydata(current_line.get_ydata())
                    if debug:
                        plt.show()
                        raw_input("Press Enter to continue...")

        if len(solutions) > 0:
            if debug:
                for a,b,c in solutions:
                    plt.plot([c/a, c/a-b], [0, a], 'r')
            print N, np.sqrt(best_d[0].dot(best_d[0])), best_d[1]
            return solutions
        N += 1
    # print np.sqrt(best_d[0].dot(best_d[0])), best_d[1]
    # raise Exception("Max order reached")
    return solutions


def h(Qx, Qy, M, tol=1e-3):
    """
    Arguments:
        Qx, Qy: 2D point to approximate
        M: max order
    """

    bounds = {}

    N = 1
    solutions = []
    best_d = [np.array([1,1]), None]

    while N <= M:
        for h,k in fareySequence(N,1):
            if 0 in (h,k):
                continue

            if (h,k) not in bounds:
                bounds[(h,k)] = {'left':np.array([-1,0]), 'right':np.array([1,0])}

            left = bounds[(h,k)]['left']
            right = bounds[(h,k)]['right']

            for x,y in resonanceSequence(N, k):

                # avoid 0-solutions
                if 0 in (x,y):
                    continue

                norm = np.sqrt(x**2+y**2)

                if h/k >= Qx:        # approach from the right
                    a, b = x, y
                else:                # approach from the left
                    a, b = x,-y

                n = np.array([ -b/norm, a/norm])


                if n.dot(np.array([-left[1], left[0]])) < 0 and \
                   n.dot(np.array([-right[1], right[0]])) > 0:

                    # nomenclature inspired in http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation
                    av = np.array([h/k-Qx, -Qy])
                    tmp = n.dot(av)
                    d = av-tmp*n
                    if d.dot(d) < best_d[0].dot(best_d[0]):
                        best_d[0] = d
                        best_d[1] = ((a, b, h*a/k))

                    if np.sqrt(d.dot(d)) <= tol:
                        ## DON'T RETURN IMMEDIATELY; THERE MIGHT BE OTHER SOLUTIONS OF THE SAME ORDER
                        # return (a, b, h*a/k)
                        solutions.append((a, b, h*a/k))
                    # the following implicitly assumes we are in the positive quadrant
                    elif d[0] < 0:
                        left = bounds[(h,k)]['left'] = n
                    else:
                        right = bounds[(h,k)]['right'] = n

        if len(solutions) > 0:
            # print np.sqrt(best_d[0].dot(best_d[0])), best_d[1]
            return solutions
        N += 1
    # print np.sqrt(best_d[0].dot(best_d[0])), best_d[1]
    # raise Exception("Max order reached")
    return solutions



def findAllMonomials(points, N, tol=1e-3, return_lowest_only=True):
    """
    Arguments:
        points: 2D (L x 2) points to approximate
        N: max order
    """
    L = points.shape[0]
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

            # apn = np.sum(n*ap, 1, keepdims=True)
            apn = np.sum(n*ap, 1, keepdims=True)
            d = ap - apn*n

            ## DON'T RETURN IMMEDIATELY; THERE MIGHT BE OTHER SOLUTIONS OF THE SAME ORDER
            indices, = np.nonzero(np.sqrt(np.sum(d*d,1)) <= tol)
            for i in indices:
                if points[i,0] >= h/k:
                    solutions[i].add((x,-y, h*x/k))
                else:
                    solutions[i].add((x, y, h*x/k))

    if return_lowest_only:
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



def monomialsForVectors(f1, f2, N, tol=1e-3):
    """
    Arguments:
        f1 (np.array_like): first frequency vector
        f2 (np.array_like): second frequency vector
        N: max order
    """
    from time import time
    st = time()

    f1 = f1.astype(np.float32)
    f2 = f2.astype(np.float32)
    F1, F2 = len(f1), len(f2)

    cart_idx = cartesian((np.arange(F1, dtype=np.uint16),
                          np.arange(F1, dtype=np.uint16),
                          np.arange(F2, dtype=np.uint16)))

    # we care only when y2 > y1
    cart_idx = cart_idx[cart_idx[:,1]>cart_idx[:,0]]

    # actual frequency triplets
    cart = np.vstack((f1[cart_idx[:,0]], f1[cart_idx[:,1]], f2[cart_idx[:,2]])).T
    nr, _ = cart_idx.shape

    # sort in order to get a*x+b*y=c with 0<x,y<1
    sorted_idx = np.argsort(cart, axis=1)
    print("c) Elapsed: {} secs".format(time() - st))
    all_points = np.zeros((nr, 2), dtype=np.float32)
    all_points[:,0] = cart[xrange(nr),sorted_idx[:,0]] / cart[xrange(nr),sorted_idx[:,2]]
    all_points[:,1] = cart[xrange(nr),sorted_idx[:,1]] / cart[xrange(nr),sorted_idx[:,2]]
    print("d) Elapsed: {} secs".format(time() - st))

    redundancy_map = defaultdict(list)
    for i,(a,b) in enumerate(all_points.tolist()):
        redundancy_map[(a,b)].append(i)
    print("e) Elapsed: {} secs".format(time() - st))

    points = np.array([[a,b] for a,b in redundancy_map])
    print("f) Elapsed: {} secs".format(time() - st))

    exponents = findAllMonomials(points, N, tol=tol)
    print("g) Elapsed: {} secs".format(time() - st))

    final_map = {}
    for k in exponents:
        sols = exponents[k]
        x, y = points[k,0], points[k,1]
        all_points_idx = redundancy_map[(x,y)]
        for idx in all_points_idx:
            key = (cart_idx[idx, 0], cart_idx[idx, 1], cart_idx[idx, 2])
            final_map[key] = np.zeros((len(sols),3), dtype=np.int16)
            for i, s in enumerate(sols):
                reordered = (sorted_idx[idx,0], sorted_idx[idx,1], sorted_idx[idx,2])
                if reordered == (0,1,2):
                    final_map[key][i,reordered] = [s[0], s[1], s[2]]
                elif reordered == (0,2,1):
                    final_map[key][i,reordered] = [-s[0], s[1], s[2]]
                elif reordered == (2,0,1):
                    final_map[key][i,reordered] = [s[0], -s[1], s[2]]
                else:
                    raise Exception("Unimplemented order!")
    print("h) Elapsed: {} secs".format(time() - st))

    # for k in final_map:
    #     f11, f12, f22 = f1[k[0]], f1[k[1]], f2[k[2]]
    #     for i in range(final_map[k].shape[0]):
    #         n1, n2, d = final_map[k][i,:].tolist()
    #         print("{}, {}, {}    {: 3d},{: 3d},{: 3d}   ->   {:.1f}*{: 3d} + {:.1f}*{: 3d} - {:.1f}*{: 3d} = {: 4.1f}".format(k[0], k[1], k[2], n1, n2, d, f11, n1, f12, n2, f22, d, f11*n1+f12*n2-f22*d))

    return final_map


if __name__ == '__main__':
    from time import time
    import pprint
    pp = pprint.PrettyPrinter(indent=4)

    max_order = 16
    tol = 1e-3

    points = np.random.rand(100,2)
    # points = np.array([[ 0.20127775,  0.39311277]])

    num_points = points.shape[0]

    tmp0 = defaultdict(set)
    st = time()
    for i in range(num_points):
        tmp = h(points[i,0], points[i,1], max_order, tol=tol)
        if len(tmp) > 0:
            tmp0[i] = set(tmp)
    print("h took {} secs (found {:.2f}%)".format(time()-st, 100*len(tmp0)/num_points))

    # st = time()
    # tmp1 = findMonomials(points, max_order, tol=tol)
    # print("findMonomials took {} secs (found {:.2f}%)".format(time()-st, 100*len(tmp1)/num_points))

    st = time()
    tmp2 = findAllMonomials(points, max_order, tol=tol)
    print("findAllMonomials took {} secs (found {:.2f}%)".format(time()-st, 100*len(tmp2)/num_points))

    # pp.pprint(points)
    # pp.pprint(tmp0)
    # pp.pprint(tmp1)
    # pp.pprint(tmp2)

    # for k in tmp1:
    #     px, py = points[k,0], points[k,1]
    #     for a,b,c in tmp1[k]:
    #         print "{:.5f} = {} \t {:.5f} ; {:.5f} ".format(a*px+b*py, c, (a*px+b*py)/c, a*px+b*py-c)

    clean = True
    for k in tmp0:
        if len(tmp0[k]-tmp2[k]) != 0:
            clean = False
            print("Error in index {}".format(k))
            print(tmp0[k])
            print(tmp2[k])

    if clean:
        print("SUCCESS!!!")