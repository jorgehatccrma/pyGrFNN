
from __future__ import division
import math

from pygrfnn.utils import memoize


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



from mpl_toolkits.mplot3d import Axes3D


import numpy as np



# got from http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


# # n1range = range(-16,17)
# # n2range = range(-16,17)
# # d_range = range(1,5)
# n1range = range(-5,6)
# n2range = range(-5,7)
# d_range = range(1,5)

# f0, f1, f2 = 2.0, 0.5, 3.0

# lattice = cartesian((n1range, n2range, d_range))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(lattice[:,0], lattice[:,1], lattice[:,2])

# ax.set_xlabel('n1')
# ax.set_ylabel('n2')
# ax.set_zlabel('d')

# plt.show()




def resonanceSequence(N, k):
    """
    Compute resonance sequence

    Arguments:
        - N (int): Order
        - k (int): denominator
    """

    pq = fareySequence(N, k)
    return [(k*p, q - k*p) for p, q in pq]


def resonancePoints(N):
    from collections import defaultdict
    from numpy.linalg import solve
    from numpy.linalg import LinAlgError

    lines = set([])
    points = defaultdict(list)

    def findIntersections(m,c):
        for ml,cl in lines:
            try:
                p = solve([[-m, 1],[-ml, 1]], [c, cl])
                if p[0] >= 0 and p[0] <= 1.0 and p[1] >= 0 and p[1] <= 1.0:
                    points[(p[0],p[1])].append((m,c))
                    points[(p[0],p[1])].append((ml,cl))
            except LinAlgError as e:
                # print e
                # print [[-m, 1],[-ml, 1]]
                pass


    for h, k in fareySequence(N, 1):
        for a, b in resonanceSequence(N, k):
            if b == 0:
                # m = np.inf
                continue
            else:
                m = a/b
                cp, cm = m*h/k, -m*h/k

                if (m,cp) not in lines:
                    findIntersections(m, cp)
                    lines.add((m,cp))

                if (m,cm) not in lines:
                    findIntersections(m, cm)
                    lines.add((m,cm))

                if m == 0:
                    continue

                if (-1/m,-cp/m) not in lines:
                    findIntersections(-1/m, -cp/m)
                    lines.add((-1/m,-cp/m))

                if (-1/m,-cm/m) not in lines:
                    findIntersections(-1/m, -cm/m)
                    lines.add((-1/m,-cm/m))

    return points




def plotResonanceDiagram(N, exclude_inf=True):
    import matplotlib.pyplot as plt

    A = 0.2

    plt.figure()
    ticks = set([])
    for h, k in fareySequence(N, 1):
        ticks.add((h,k))
        for a, b in resonanceSequence(N, k):
            if b == 0:
                if not exclude_inf:
                    plt.plot([h/k, h/k], [0, 1], 'b:', alpha=2*A)
                    plt.plot([0, 1], [h/k, h/k], 'b:', alpha=2*A)
                continue
            m = a/b
            cp, cm = m*h/k, -m*h/k
            x = np.array([0, h/k, 1])
            y = np.array([cp, 0, cm+m])
            plt.plot(  x,   y, 'b', alpha=A)
            plt.plot(  y,   x, 'b', alpha=A)
            plt.plot(  x, 1-y, 'b', alpha=A)
            plt.plot(1-y,   x, 'b', alpha=A)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([h/k for h,k in ticks], [r"$\frac{{{:d}}}{{{:d}}}$".format(h,k) for h,k in ticks])
    plt.yticks([h/k for h,k in ticks], [r"$\frac{{{:d}}}{{{:d}}}$".format(h,k) for h,k in ticks])
    # plt.xticks([h/k for h,k in ticks], [r"${:d}/{:d}$".format(h,k) for h,k in ticks])
    # plt.yticks([h/k for h,k in ticks], [r"${:d}/{:d}$".format(h,k) for h,k in ticks])
    plt.title("N = {:d}".format(N))



# def h(Qx, Qy, M, tol=1e-3, debug=False):
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
#         pt_hk, = plt.plot(0, 0, 'mo')
#         lines = {}
#         plt.xlim(0,1)
#         plt.ylim(0,1)

#     N = 1
#     candidates = []
#     best_d = [1, None]

#     if debug:
#         print("Searching for (Qx,Qy)=({},{})".format(Qx,Qy))

#     while N <= M:
#         # if debug:
#         #     print("N={}".format(N))
#         for h,k in fareySequence(N,1):
#             if h == 0 or k == 0:
#                 # if debug:
#                 #     print("skipping h/k={}/{}".format(h,k))
#                 continue
#             if debug:
#                 # print("h/k={}/{}".format(h,k))
#                 pt_hk.set_xdata(h/k);
#             if (h,k) not in bounds:
#                 # bounds[(h,k)] = [(0,1), (1,0), (0,1), (1,0)]
#                 bounds[(h,k)] = [(0,1), (1,1e-32), (0,1), (1,1e-32)]  # horrible hack to avoid division by zero errors
#                 if debug:
#                     lines[(h,k)] = {}
#                     lines[(h,k)]['lr_line'], = plt.plot(None, None, 'g')
#                     lines[(h,k)]['ur_line'], = plt.plot(None, None, 'g:')
#                     lines[(h,k)]['ll_line'], = plt.plot(None, None, 'b-')
#                     lines[(h,k)]['ul_line'], = plt.plot(None, None, 'b:')

#             # TODO: use binary search for speedup?
#             ll, ul, lr, ur = (p/q for p,q in bounds[(h,k)])
#             for x,y in resonanceSequence(N, k):
#                 if h/k >= Qx:        # approach from the right
#                     if x > 0 and y > 0 and x/y > lr and x/y < ur:
#                         if debug:
#                             current_line.set_xdata([h/k, 0])
#                             current_line.set_ydata([0, (x/y)*(h/k)])
#                         d = Qx*x + Qy*y - h*x/k
#                         if np.abs(d) < np.abs(best_d[0]):
#                             best_d[0] = d
#                             best_d[1] = ((x, y, h*x/k))
#                         if np.abs(d) <= tol:
#                             # return (x, y, h*x/k)
#                             candidates.append((x, y, h*x/k))
#                         elif d > 0:
#                             bounds[(h,k)][2] = (x,y)
#                             lr = x/y
#                             if debug:
#                                 lines[(h,k)]['lr_line'].set_xdata(current_line.get_xdata())
#                                 lines[(h,k)]['lr_line'].set_ydata(current_line.get_ydata())
#                         else:
#                             bounds[(h,k)][3] = (x,y)
#                             ur = x/y
#                             if debug:
#                                 lines[(h,k)]['ur_line'].set_xdata(current_line.get_xdata())
#                                 lines[(h,k)]['ur_line'].set_ydata(current_line.get_ydata())
#                         if debug:
#                             plt.show()
#                             # raw_input("Press Enter to continue...")
#                 else: # approach from the left
#                     if x > 0 and y > 0 and x/y > ll and x/y < ul:
#                         if debug:
#                             current_line.set_xdata([h/k, 1])
#                             current_line.set_ydata([0, (x/y)*(1-(h/k))])
#                         d = Qx*x - Qy*y - h*x/k
#                         if np.abs(d) < np.abs(best_d[0]):
#                             best_d[0] = d
#                             best_d[1] = ((x, -y, h*x/k))
#                         if np.abs(d) <= tol:
#                             # return (x, -y, h*x/k)
#                             candidates.append((x, -y, h*x/k))
#                         elif d < 0:
#                             bounds[(h,k)][0] = (x,y)
#                             ll = x/y
#                             if debug:
#                                 lines[(h,k)]['ll_line'].set_xdata(current_line.get_xdata())
#                                 lines[(h,k)]['ll_line'].set_ydata(current_line.get_ydata())
#                         else:
#                             bounds[(h,k)][1] = (x,y)
#                             ul = x/y
#                             if debug:
#                                 lines[(h,k)]['ul_line'].set_xdata(current_line.get_xdata())
#                                 lines[(h,k)]['ul_line'].set_ydata(current_line.get_ydata())
#                         if debug:
#                             plt.show()
#                             # raw_input("Press Enter to continue...")
#             if len(candidates) > 0:
#                 print best_d
#                 return candidates
#         N += 1
#     print best_d
#     raise Exception("Max order reached")


def h(Qx, Qy, M, tol=1e-3, debug=False):
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
        pt_hk, = plt.plot(0, 0, 'mo')
        lines = {}
        plt.xlim(0,1)
        plt.ylim(0,1)

    N = 1
    candidates = []
    best_d = [np.array([1,1]), None]

    if debug:
        print("Searching for (Qx,Qy)=({},{})".format(Qx,Qy))

    while N <= M:
        # if debug:
        #     print("N={}".format(N))
        for h,k in fareySequence(N,1):
            if h == 0 or k == 0:
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

                if 0 in [x,y]:
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

                    # nomenclature inspired in http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation
                    av = np.array([h/k-Qx, -Qy])
                    tmp = n.dot(av)
                    d = av-tmp*n
                    if d.dot(d) < best_d[0].dot(best_d[0]):
                        best_d[0] = d
                        best_d[1] = ((a, b, h*a/k))

                    if np.sqrt(d.dot(d)) <= tol:
                        # return (a, b, h*a/k)
                        candidates.append((a, b, h*a/k))
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
                        # raw_input("Press Enter to continue...")

            if len(candidates) > 0:
                if debug:
                    for a,b,c in candidates:
                        plt.plot([c/a, c/a-b], [0, a], 'r')
                print np.sqrt(best_d[0].dot(best_d[0])), best_d[1]
                return candidates
        N += 1
    print np.sqrt(best_d[0].dot(best_d[0])), best_d[1]
    # raise Exception("Max order reached")
    return candidates



# Test values
# (1/4,1/3) -> (4,3,2)
# (1/4,1/4) -> (2,2,1) or (3,1,1)