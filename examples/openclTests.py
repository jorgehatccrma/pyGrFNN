import pyopencl as cl

platforms = cl.get_platforms()

for p in platforms:
    print p
