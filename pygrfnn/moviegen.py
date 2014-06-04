from __future__ import print_function
import sys
from tempfile import NamedTemporaryFile as TmpFile
import shutil
from matplotlib import animation
import matplotlib.pyplot as plt
from utils import find_nearest
from vis import tf_detail
import numpy as np
from scipy.io.wavfile import write
from subprocess import call


def animate_tf_detail(TF, ods, t_ods, sr_ods, T, signal, outfile, fps=24):

    # First set up the figure, the axis, and the plot element we want to animate
    (fig, tf_line, t_line, detail) = tf_detail(TF, t_ods, T, np.max(t_ods)/2.0, ods, np.abs)

    # initialization function
    def init():
        tf_line.set_xdata([0, 0])
        t_line.set_xdata([0, 0])
        detail.set_xdata(np.abs(TF[:,0]))
        return tf_line, t_line, detail

    # animation function.  This is called sequentially
    def animate(i):
        t_detail = 1.0*i/fps
        __, detail_idx = find_nearest(t_ods, t_detail)
        tf_line.set_xdata([t_detail, t_detail])
        t_line.set_xdata([t_detail, t_detail])
        detail.set_xdata(np.abs(TF[:,detail_idx]))

        print('  {0:.2f}/{1:.2f}'.format(t_detail, max(t_ods)), end='\r')
        sys.stdout.flush()

        return tf_line, t_line, detail

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig,
                                   animate,
                                   init_func=init,
                                   frames=int(fps*np.max(t_ods))+1, # +1?
                                   interval=1000.0/fps,
                                   # blit=True)
                                   blit=False)

    with TmpFile(suffix='.mp4') as temp_video, TmpFile(suffix='.wav') as temp_audio:
        # FFwriter = animation.FFMpegWriter(fps=fps, bitrate=40000)
        FFwriter = animation.FFMpegWriter(fps=fps, bitrate=10000)
        anim.save(temp_video.name, writer=FFwriter, extra_args=['-vcodec', 'libx264', '-preset', 'slow'])

        # add audio
        write(temp_audio.name, signal.sr, signal.data)
        call(['ffmpeg', '-i', temp_video.name,
                        # '-itsoffset', '00:00:00.1',  # for some reason, the first bit of audio is lost if not delayed
                        '-i', temp_audio.name,
                        '-vcodec', 'copy',
                        '-acodec', 'aac',
                        '-strict', 'experimental',
                        '-y',  # overwrite output file w/o asking
                        outfile])

