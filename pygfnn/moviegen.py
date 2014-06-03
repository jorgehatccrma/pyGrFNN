from tempfile import NamedTemporaryFile as TmpFile
import shutil
from matplotlib import animation
import matplotlib.pyplot as plt
from utils import find_nearest
from vis import tf_detail
import numpy as np
from scipy.io.wavfile import write
from subprocess import call


# #################################################
# # UTILS TO GENERATE AND EMBED THE MOVIE
# #################################################


# WEBM_VIDEO_TAG = """<video controls>
#  <source src="data:video/x-webm;base64,{0}" type="video/webm">
#  Your browser does not support the video tag.
# </video>"""

# M4V_VIDEO_TAG = """<video controls>
#  <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
#  Your browser does not support the video tag.
# </video>"""

# FPS = 24         # Frames per second in the generated movie

# def anim_to_html(anim, filename=None):
#     if not hasattr(anim, '_encoded_video'):
#         with NamedTemporaryFile(suffix='.webm') as f:
#             # webm_writer = animation.FFMpegWriter(fps=FPS, codec="libvpx")  # you'll need libvpx to encode .webm videos
#             webm_writer = animation.FFMpegFileWriter(fps=FPS, codec="libvpx")  # you'll need libvpx to encode .webm videos

#             # # H.264
#             # webm_writer = animation.FFMpegFileWriter(fps=FPS, codec="libx264")  # you'll need libvpx to encode .webm videos

#             vpx_args = ["-quality", "good",    # many arguments are not needed in this example, but I left them for reference
#                         "-cpu-used", "0",
#                         "-b:v", "500k",
#                         "-qmin", "10",
#                         "-qmax", "42",
#                         "-maxrate", "500k",
#                         "-bufsize", "1000k",
#                         "-threads", "4",
#                         "-vf", "scale=-1:240",
#                         # "-codec:a", "libvorbis",
#                         # "-b:a", "128k",
#                         ]
#             anim.save(f.name, writer=webm_writer, extra_args=vpx_args)
#             if filename is not None:  # in case you want to keep a copy of the generated movie
#                 shutil.copyfile(f.name, filename)
#             video = open(f.name, "rb").read()
#         anim._encoded_video = video.encode("base64")

#     return WEBM_VIDEO_TAG.format(anim._encoded_video)


# from IPython.display import HTML

# def display_animation(anim, filename):
#     plt.close(anim._fig)
#     return HTML(anim_to_html(anim, filename))



#################################################
# THE ACTUAL ANIMATION
#################################################


def animate_tf_detail(TF, ods, t_ods, sr_ods, T, signal, outfile, fps=24):

    plt.ioff()

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
        _, detail_idx = find_nearest(t_ods, t_detail)
        tf_line.set_xdata([t_detail, t_detail])
        t_line.set_xdata([t_detail, t_detail])
        detail.set_xdata(np.abs(TF[:,detail_idx]))
        return tf_line, t_line, detail

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init,
                                   frames=int(fps*np.max(t_ods))+1, # +1?
                                   interval=1000.0/fps,
                                   # blit=True)
                                   blit=False)

    with TmpFile(suffix='.mp4') as temp_video, TmpFile(suffix='.wav') as temp_audio:
        # FFwriter = animation.FFMpegWriter(fps=fps, bitrate=40000)
        FFwriter = animation.FFMpegWriter(fps=fps, bitrate=10000)
        anim.save(temp_video.name, writer=FFwriter, extra_args=['-vcodec', 'libx264', '-preset', 'ultrafast'])

        # add audio
        write(temp_audio.name, signal.sr, signal.data)
        call(['ffmpeg', '-i', temp_video.name,
                        # '-itsoffset', '00:00:00.1',  # for some reason, the first bit of audio is lost if not delayed
                        '-i', temp_audio.name,
                        '-vcodec', 'copy',
                        '-acodec', 'aac',
                        '-strict', 'experimental',
                        outfile])

    plt.ion()
