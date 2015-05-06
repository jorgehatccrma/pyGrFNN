"""Plotting functions

Note:
    It relies on Matplotlib

"""

import warnings

import logging
logger = logging.getLogger('pygrfnn.vis')

import numpy as np
from functools import wraps
from numpy.polynomial.polynomial import polyroots

from utils import find_nearest
from utils import nice_log_values
from grfnn import grfnn_update_event

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    warnings.warn("Failed to import matplotlib. Plotting functions will be disabled.")



# graphical output decorator
def check_display(fun):
    """Decorator to check for display capability

    Args:
        fun (function): Function to be timed

    Returns:
        (function): decorated function
    """
    @wraps(fun)
    def display_wrapper(*args, **kwargs):
        try:
            import matplotlib as mpl
            import os
            if "DISPLAY" in os.environ:
                return fun(*args, **kwargs)
            else:
                warnings.warn("Couldn't find a DISPLAY, so visualizations are disabled")
                # logging.info("Couldn't find a DISPLAY, so visualizations are disabled")
                # FIXME: we could use Agg backend and save to file
                # (see http://stackoverflow.com/questions/8257385/automatic-detection-of-display-availability-with-matplotlib)
        except ImportError:
            warnings.warn("Couldn't find a DISPLAY, so visualizations are disabled")
            # logging.info("Couldn't import matplotlib, so visualizations are disabled")

    return display_wrapper



@check_display
def tf_simple(TF, t, f, title=None, x=None, display_op=np.abs,
              cmap='binary', vmin=None, vmax=None):
    """tf_simple(TF, t, f, x=None, display_op=np.abs)

    Simple time-frequency representation. It shows the TF in the top plot and
    the original time signal in the bottom plot, is specified.

    Args:
        TF  (:class:`numpy.array`): time-frequency representation
        t (:class:`numpy.array`): time vector
        f (:class:`numpy.array`): frequency vector
        title (string): title of the plot
        x (:class:`numpy.array`): original time domain signal. If *None*, no
            time domain plot is shown
        display_op (function): operator to apply to the TF representation (e.g.
            `numpy.abs`)
        cmap (`string`): colormap to use in the TF representation
        vmin (float): if not `None`, defines the lower limit of the colormap
        vmax (float): if not `None`, defines the upper limit of the colormap

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
                               height_ratios=[3, 1]
                               )
        gs.update(wspace=0.0, hspace=0.0)  # set the spacing between axes.
        axTF = fig.add_subplot(gs[0])
        axOnset = fig.add_subplot(gs[1], sharex=axTF)

    axTF.pcolormesh(t, f, opTF, cmap=cmap, vmin=vmin, vmax=vmax)

    if title is not None:
        axTF.set_title(title)

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


@check_display
def tf_detail(TF, t, f, title=None, t_detail=None, x=None, display_op=np.abs,
              figsize=None, cmap='binary', vmin=None, vmax=None):
    """tf_detail(TF, t, f, t_detail=None, x=None, display_op=np.abs)

    Detailed time-frequency representation. It shows the TF in the top plot. It
    also shows the frequency representation at a specific time (last time by
    default) on the plot at the right. If specified, the original time signal
    is shown the bottom plot.

    Args:
        TF (:class:`numpy.array`): time-frequency representation
        t (:class:`numpy.array`): time vector
        f (:class:`numpy.array`): frequency vector
        title (string): title of the plot
        t_detail (float or list of floats): time instant(s) to be detailed
        x (:class:`numpy.array`): original time domain signal. If *None*, not
            time domain plot is shown
        display_op (function): operator to apply to the TF representation (e.g.
            `numpy.abs`)
        figsize (tuple, optional): matplotlib's figure size
        cmap (`string`): colormap to use in the TF representation
        vmin (float): if not `None`, defines the lower limit of the colormap
        vmax (float): if not `None`, defines the upper limit of the colormap

    Returns:
        (handles, ...): tuple of handles to plotted elements. They can be used
            to create animations

    Note:
        `vmin` and `vmax` are useful when comparing different time-frequency
        representations, so thay all share the same color scale.


    Note:
        Is responsibility of the caller to issue the ``plt.show()`` command if
        necessary

    """
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    opTF = display_op(TF)

    if t_detail is None:
        wr = [1, 2, 20]
        detail = None
    else:
        wr = [1, 2, 20, 6]

    if x is None:
        hr = [1]
        axOnset = None
    else:
        hr = [3, 1]

    gs = gridspec.GridSpec(len(hr), len(wr),
                           width_ratios=wr,
                           height_ratios=hr
                           )
    gs.update(wspace=0.0, hspace=0.0)  # set the spacing between axes.


    axCB = fig.add_subplot(gs[0])
    axTF = fig.add_subplot(gs[2])

    if x is not None:
        axOnset = fig.add_subplot(gs[len(wr)+2], sharex=axTF)

    if t_detail is not None:
        axF = fig.add_subplot(gs[3], sharey=axTF)

    nice_freqs = nice_log_values(f)

    # TF image
    # im = axTF.pcolormesh(t, f, opTF, cmap=cmap)
    im = axTF.imshow(opTF,
                     extent=[min(t), max(t), min(f), max(f)],
                     cmap=cmap,
                     vmin=vmin,
                     vmax=vmax,
                     origin='lower'
                     )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        axTF.set_yscale('log')
        axTF.set_yticks(nice_freqs)
        axTF.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        axTF.invert_yaxis()

    if title is not None:
        axTF.set_title(title)

    # Add colorbar
    cb = plt.colorbar(im, ax=axTF, cax=axCB)
    cb.ax.yaxis.set_ticks_position('left')

    # TF detail
    # find detail index
    tf_line = None
    tf_x_min, tf_x_max = 0, np.max(opTF)
    if vmin is not None:
        tf_x_min = vmin
    if vmax is not None:
        tf_x_max = vmax

    if t_detail is not None:
        if isinstance(t_detail, np.ndarray):
            t_detail = t_detail.tolist()
        elif not isinstance(t_detail, list):
            t_detail = [t_detail]
        t_detail, idx = find_nearest(t, t_detail)
        # axF.invert_xaxis()
        detail = axF.semilogy(opTF[:, idx], f)
        axF.set_yticks(nice_freqs)
        axF.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        axF.xaxis.set_ticks_position('top')
        axF.axis('tight')
        axF.set_xlim(tf_x_min, tf_x_max)
        axF.yaxis.set_ticks_position('right')
        plt.setp(axF.get_xaxis().get_ticklabels(), rotation=-90 )
        axTF.hold(True)
        tf_line = axTF.plot([t_detail, t_detail], [np.min(f), np.max(f)])
        # tf_line = [axTF.axvline(td) for td in t_detail]
        axTF.hold(False)
    axTF.axis('tight')


    # onset signal
    t_line = None
    if axOnset is not None:
        plt.setp(axTF.get_xticklabels(), visible=False)
        axOnset.plot(t, x, color='k')
        if t_detail is not None:
            t_line = axOnset.plot([t_detail, t_detail], [np.min(x), np.max(x)])
            # t_line = [axOnset.axvline(td) for td in t_detail]
        axOnset.yaxis.set_ticks_position('right')
        axOnset.axis('tight')

    # plt.show()

    return (fig, im, tf_line, t_line, detail)


@check_display
def plot_connections(connection, title=None, f_detail=None, display_op=np.abs,
                     detail_type='polar', cmap='binary', vmin=None, vmax=None):
    """plot_connections(connection, t_detail=None, display_op=np.abs,
                        detail_type='polar')

    Args:
        connection (:class:`.Connection`): connection object
        title (string): Title to be displayed
        f_detail (float): frequency of the detail plot
        display_op (function): operator to apply to the connection
            matrix (e.g. `numpy.abs`)
        detail_type (string): detail complex display type ('cartesian',
            'polar', 'magnitude' or 'phase')
        cmap (`string`): colormap to use in the TF representation
        vmin (float): if not `None`, defines the lower limit of the colormap
        vmax (float): if not `None`, defines the upper limit of the colormap

    Note:
        Is responsibility of the caller to issue the ``plt.show()``
        command if necessary

    """

    fig = plt.figure()

    if f_detail is not None:
        gs = gridspec.GridSpec(2, 1,
                               width_ratios=[1],
                               height_ratios=[3, 1]
                               )
        gs.update(wspace=0.0, hspace=0.0)  # set the spacing between axes.
        axConn = fig.add_subplot(gs[0])
        axDetail = fig.add_subplot(gs[1])
    else:
        axConn = fig.add_subplot(1, 1, 1)

    f_source = connection.source.f
    f_dest = connection.destination.f
    matrix = connection.matrix
    opMat = display_op(matrix)

    # axConn.pcolormesh(f_source, f_dest, opMat, cmap=cmap)
    axConn.imshow(opMat,
                     extent=[min(f_source), max(f_source),
                             min(f_dest), max(f_dest)],
                     cmap=cmap,
                     vmin=vmin,
                     vmax=vmax,
                     origin='lower'
                     )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # axConn.invert_yaxis()
        axConn.set_xscale('log')
        axConn.set_xticks(nice_log_values(f_source))
        axConn.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        axConn.set_yscale('log')
        axConn.set_yticks(nice_log_values(f_dest))
        axConn.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        axConn.set_ylabel(r'$f_{\mathrm{dest}}$')

    if title is not None:
        axConn.set_title(title)

    if f_detail is None:
        axConn.set_xlabel(r'$f_{\mathrm{source}}$')
    else:
        (f_detail, idx) = find_nearest(f_dest, f_detail)
        conn = matrix[idx, :]

        axConn.hold(True)
        axConn.plot([np.min(f_source), np.max(f_source)],
                    [f_detail, f_detail],
                    color='r')
        axConn.hold(False)

        scalar_formatter = mpl.ticker.ScalarFormatter()

        if detail_type is 'polar':
            axDetail.semilogx(f_source, np.abs(conn))
            axDetailb = axDetail.twinx()
            axDetailb.semilogx(f_source, np.angle(conn), color='r')
            axDetailb.set_xticks(nice_log_values(f_source))
            axDetailb.get_xaxis().set_major_formatter(scalar_formatter)
            axDetailb.set_ylim([-np.pi, np.pi])
            axDetail.axis('tight')
        elif detail_type is 'magnitude':
            y_min, y_max = 0, np.abs(conn)
            if vmin is not None:
                y_min = vmin
            if vmax is not None:
                y_max = vmax
            axDetail.semilogx(f_source, np.abs(conn))
            axDetail.set_xticks(nice_log_values(f_source))
            axDetail.get_xaxis().set_major_formatter(scalar_formatter)
            # axDetail.axis('tight')
            axDetail.set_ylim([y_min, y_max])
        elif detail_type is 'phase':
            axDetail.semilogx(f_source, np.angle(conn), color='r')
            axDetail.set_xticks(nice_log_values(f_source))
            axDetail.get_xaxis().set_major_formatter(scalar_formatter)
            axDetail.set_ylim([-np.pi, np.pi])
        else:
            axDetail.semilogx(f_source, np.real(conn))
            axDetailb = axDetail.twinx()
            axDetailb.semilogx(f_source, np.imag(conn), color='r')
            axDetailb.set_xticks(nice_log_values(f_source))
            axDetailb.get_xaxis().set_major_formatter(scalar_formatter)
            axDetail.axis('tight')
        axDetail.set_xlabel(r'$f_{\mathrm{dest}}$')

        axConn.set(aspect=1, adjustable='box-forced')

    # plt.show()


@check_display
class GrFNN_RT_plot(object):
    """
    On-line GrFNN state visualization.

    Args:
        grfnn (:class:`.network.Model`): GrFNN to be plotted
        update_interval (float): Update interval (in seconds). This is
            an approximation, as the update will happen as a multiple of the
            integration step time.
        fig_name (string): Name of the figure to use. If specified, the same
            figure will be reused in consecutive runs. If None, a new figure
            will be created each time the caller script runs.
        title (string): optional title of the plot

    Note:
        This function calls ``plt.ion()`` internally to allow for on-line
        updating of the plot

    Note:
        There is probably room for optimization here. For example,
        http://bastibe.de/2013-05-30-speeding-up-matplotlib.html does some
        interesting analysis/optimizations for updating plots
    """

    def __init__(self, grfnn, update_interval=0, fig_name=None, title=''):
        self.grfnn = grfnn
        self.update_interval = update_interval
        self.title = title
        self.fig_name = fig_name
        plt.ion()
        if fig_name is None:
            self.fig = plt.figure()
        else:
            self.fig = plt.figure(fig_name)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)
        self.line1, = self.ax.semilogx(grfnn.f, np.abs(grfnn.z), 'k')
        self.ax.axis((np.min(grfnn.f), np.max(grfnn.f), 0, 1))
        plt.xticks(nice_log_values(grfnn.f))
        self.ax.set_title('{}'.format(self.title))
        self.fig.canvas.draw()

        self.last_update = 0

        def update_callback(sender, **kwargs):
            """
            Update the plot when necessary
            """
            t = kwargs['t']

            if 'force' in kwargs:
                force = kwargs['force']
            else:
                force = False

            if force or (t - self.last_update >= self.update_interval):
                z = sender.z
                self.line1.set_ydata(np.abs(z))
                self.ax.set_title('{} (t = {:0.2f}s)'.format(self.title, t))
                self.fig.canvas.draw()
                self.last_update = t

        grfnn_update_event.connect(update_callback, sender=grfnn, weak=False)


@check_display
def vector_field(params, F=1.0):
    """
    Args:
        params (`.Zparam`): oscillator's intrinsic parameters
        F (scalar or array_like): Forcing values to plot

    ToDo:
        Add reference
    """
    colormap = plt.cm.gist_heat

    try:
        len(F)
    except:
        F = [F]

    # FIXME: customizable?
    colors = [colormap(i) for i in np.linspace(0, 0.7, len(F))]

    # \dot{r} = f(r, F)
    r = np.arange(0, 1/np.sqrt(params.epsilon), 0.01)
    rdot = np.add.outer(params.alpha * r +
                        params.beta1 * r**3 +
                        ((params.epsilon* params.beta2 * r**5) /
                            (1 - params.epsilon * r**2)),
                        F)

    # plot it
    plt.figure()
    ax = plt.gca()
    ax.set_color_cycle(colors)
    plt.plot(r, rdot, zorder=0, linewidth=2)
    plt.title(r'$\alpha={:.3g},'
              r'\beta_1={:.3g},'
              r'\beta_2={:.3g}$'.format(params.alpha,
                                        params.beta1,
                                        params.beta2))

    ## assymptote
    # plt.vlines(x=1/np.sqrt(epsilon), ymin=-1, ymax=2, color='r', linestyle=':')
    # plt.ylim(-5,5)
    ax.axhline(y=0,xmin=min(r),xmax=max(r),c="k",zorder=5, alpha=0.5)

    plt.xlabel(r'$r$')
    plt.ylabel(r'$\dot{r}$', labelpad=-10)

    # find roots (r^*)
    roots = [None] * len(F)
    for i in xrange(len(F)):
        r = polyroots([F[i],  # ^0
                       params.alpha,  # ^1
                       -params.epsilon*F[i],  # ^2
                       params.beta1-params.epsilon*params.alpha,  # ^3
                       0,  # ^4
                       params.epsilon*(params.beta2-params.beta1)])  # ^5
        r = np.real(r[np.abs(np.imag(r)) < 1e-20])
        r = r[(r>=0) & (r < 1/params.sqe)]
        roots[i] = r
    # print roots

    # plot the roots
    plt.gca().set_color_cycle(colors)
    for r in roots:
        plt.plot(r, np.zeros_like(r), 'o', markersize=4, zorder=10)
