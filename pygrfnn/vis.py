"""Plotting functions

Note:
    It relies on Matplotlib

"""

import numpy as np
from functools import wraps
from utils import find_nearest
from utils import nice_log_values

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)



MPL = True
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    MPL = False



# plotting decorator (checks for available matplotlib)
def check_mpl(fun):
    """Decorator to check for Matplotlib availability

    Args:
        fun (function): Plotting function

    Returns:
        (function): decorated function

    Example: ::

        from pygrfnn.vis import check_mpl

        # decorate a function
        @check_mpl
        def my_plot(x, y):
            plt.plot(x,y)


        # use it as you would normally would
        my_plot(np.arange(10), np.random.rand(10))

    """
    @wraps(fun)
    def mpl_wrapper(*args, **kwargs):
        if MPL:
            output = fun(*args, **kwargs)
        else:
            logging.info('Skipping call to %s() (couldn\'t import Matplotib or'\
                         ' one of its modules)' % (fun.__name__,))
            output = None
        return output

    return mpl_wrapper


@check_mpl
def tf_simple(TF, t, f, x=None, display_op=np.abs):
    """tf_simple(TF, t, f, x=None, display_op=np.abs)

    Simple time-frequency representation. It shows the TF in the top plot and
    the original time signal in the bottom plot, is specified.

    Args:
        TF  (:class:`numpy.array`): time-frequency representation
        t (:class:`numpy.array`): time vector
        f (:class:`numpy.array`): frequency vector
        x (:class:`numpy.array`): original time domain signal. If *None*, not
            time domain plot is shown
        display_op (function): operator to apply to the TF representation (e.g.
            `numpy.abs`)

    Note:
        Is responsibility of the caller to issue the ``plt.show()`` command if
        necessary

    """

    opTF = display_op(TF)

    if x is None:
        fig, axTF = plt.subplots(1)
        axOnset = None
    else:
        # fig, (axTF, axT) = plt.subplots(2, 1, sharex=True)
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1,
                           width_ratios=[1],
                           height_ratios=[3,1]
                           )
        gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes.
        axTF = fig.add_subplot(gs[0])
        axOnset = fig.add_subplot(gs[1], sharex=axTF)

    axTF.pcolormesh(t, f, opTF, cmap='binary')

    axTF.set_yscale('log')
    axTF.set_yticks(nice_log_values(f))
    axTF.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axTF.axis('tight')

    if axOnset is not None:
        plt.setp(axTF.get_xticklabels(), visible=False)
        axOnset.plot(t, x)
        axOnset.yaxis.set_ticks_position('right')
        axOnset.axis('tight')

    # plt.show()




@check_mpl
def tf_detail(TF, t, f, t_detail=None, x=None, display_op=np.abs):
    """tf_detail(TF, t, f, t_detail=None, x=None, display_op=np.abs)

    Detailed time-frequency representation. It shows the TF in the top plot. It
    also shows the frequency representation at a specific time (last time by
    default) on the plot at the right. If specified, the original time signal is
    shown the bottom plot.

    Args:
        TF (:class:`numpy.array`): time-frequency representation
        t (:class:`numpy.array`): time vector
        f (:class:`numpy.array`): frequency vector
        t_detail (float): time instant to be detailed
        x (:class:`numpy.array`): original time domain signal. If *None*, not
            time domain plot is shown
        display_op (function): operator to apply to the TF representation (e.g.
            `numpy.abs`)

    Returns:
        (handles, ...): tuple of handles to plotted elements. They can be used
            to create animations

    Note:
        Is responsibility of the caller to issue the ``plt.show()`` command if
        necessary

    """

    fig = plt.figure()
    opTF = display_op(TF)

    if x is None:
        gs = gridspec.GridSpec(1, 4,
                           width_ratios=[1,2,20,3],
                           height_ratios=[1]
                           )
        gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes.
        axOnset = None
    else:
        gs = gridspec.GridSpec(2, 4,
                           width_ratios=[1,2,20,3],
                           height_ratios=[3,1]
                           )
        gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes.

    axCB = fig.add_subplot(gs[0])
    axTF = fig.add_subplot(gs[2])
    axF = fig.add_subplot(gs[3], sharey=axTF)

    if x is not None:
        axOnset = fig.add_subplot(gs[6], sharex=axTF)

    # find detail index
    if t_detail is None:
        idx = -1    # last column
    else:
        t_detail, idx = find_nearest(t, t_detail)

    nice_freqs = nice_log_values(f)

    # TF image
    im = axTF.pcolormesh(t, f, opTF, cmap='binary')
    axTF.set_yscale('log')
    axTF.set_yticks(nice_freqs)
    axTF.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    tf_line, = axTF.plot([t_detail, t_detail], [np.min(f), np.max(f)], color='r')
    axTF.axis('tight')

    # Add colorbar
    cb = plt.colorbar(im, ax=axTF, cax=axCB)
    cb.ax.yaxis.set_ticks_position('left')

    # TF detail
    detail, = axF.semilogy(opTF[:,idx], f)
    axF.set_yticks(nice_freqs)
    axF.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axF.set_xticklabels([])
    # axF.set_xticks([np.min(opTF[:,idx]), np.max(opTF[:,idx])])
    axF.xaxis.set_ticks_position('top')
    axF.axis('tight')
    axF.xlim([0, np.max(opTF)])
    axF.yaxis.set_ticks_position('right')

    # onset signal
    t_line = None
    if axOnset is not None:
        plt.setp(axTF.get_xticklabels(), visible=False)
        axOnset.plot(t, x)
        t_line, = axOnset.plot([t_detail, t_detail], [np.min(x), np.max(x)], color='r')
        axOnset.yaxis.set_ticks_position('right')
        axOnset.axis('tight')

    # plt.show()

    return (fig, tf_line, t_line, detail)



@check_mpl
def plot_connections(connection, f_detail=None, display_op=np.abs,
                     detail_type='polar'):
    """plot_connections(connection, t_detail=None, display_op=np.abs,
                        detail_type='polar')

    Args:
        connection (:class:`.Connection`): connection object
        f_detail (float): frequency of the detail plot
        display_op (function): operator to apply to the connection matrix (e.g.
            `numpy.abs`)
        detail_type (string): detail complex display type ('polar' or 'cartesian')

    Note:
        Is responsibility of the caller to issue the ``plt.show()`` command if
        necessary

    """

    opMat = display_op(matrix)
    fig = plt.figure()

    if f_detail is not None:
        gs = gridspec.GridSpec(2, 1,
                       width_ratios=[1],
                       height_ratios=[3,1]
                       )
        gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes.
        axConn = fig.add_subplot(gs[0])
        axDetail = fig.add_subplot(gs[1])
    else:
        axConn = fig.add_subplot(1,1,1)

    f_source = connection.source.f
    f_dest = connection.destination.f
    matrix = connection.matrix

    axConn.pcolormesh(f_dest, f_source, opMat, cmap='binary')
    axConn.set_xscale('log')
    axConn.set_xticks(nice_log_values(f_dest))
    axConn.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axConn.set_yscale('log')
    axConn.set_yticks(nice_log_values(f_source))
    axConn.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axConn.plot([np.min(f_dest), np.max(f_dest)], [f_detail, f_detail], color='r')
    axConn.axis('tight')

    if f_detail is not None:
        (f_detail, idx) = find_nearest(f_source, f_detail)
        conn = matrix[idx, :]

        if detail_type is not 'cartesian':
            axDetail.semilogx(f_dest, np.abs(conn))
            axDetailb = axDetail.twinx()
            axDetailb.semilogx(f_dest, np.angle(conn), color='r')
            axDetailb.set_xticks(nice_log_values(f_dest))
            axDetailb.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            axDetailb.set_ylim([-np.pi, np.pi])
        else:
            axDetail.semilogx(f_dest, np.real(conn))
            axDetailb = axDetail.twinx()
            axDetailb.semilogx(f_dest, np.imag(conn), color='r')
            axDetailb.set_xticks(nice_log_values(f_dest))
            axDetailb.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        axDetail.axis('tight')

    # plt.show()


