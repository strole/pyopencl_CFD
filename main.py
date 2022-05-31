import numpy as np
import math
import time
import sys
import pyopencl as cl

np.set_printoptions(threshold=sys.maxsize)

class PP:
    def __init__(self, path):
        self.memfag = cl.mem_flags
        self.context = cl.create_some_context(interactive=True)
        self.queue = cl.CommandQueue(self.context)
        self.code = "".join(open(path, 'r').readlines())
        self.program = cl.Program(self.context, self.code).build()

    def getQueue(self):
        return self.queue

    def getProgram(self):
        return self.program

    def getFlags(self):
        return self.memfag

    def getContext(self):
        return self.context

def main():

    p = PP("./lernel.cl")

    error, bnorm, tolerance, scalefactor, numiter, printfreq = 0, 0, 0, 0, 0, 1000
    bbase, hbase, wbase, mbase, nbase = 10, 15, 5, 32, 32
    irrotational, checkerr = 1, 0
    m, n, b, h, w, iter, i, j = 0, 0, 0, 0, 0, 0, 0, 0
    tstart, tstop, ttot, titer = 0, 0, 0, 0

    print("Usage: cfd <scale> <numiter>\n")

    scalefactor, numiter = 64, 1000

    print("Scale Factor = {}, iterations = {}\n".format(scalefactor, numiter))

    b = bbase * scalefactor
    h = hbase * scalefactor
    w = wbase * scalefactor
    m = mbase * scalefactor
    n = nbase * scalefactor

    print("Running CFD on {} x {} grid in serial\n".format(m, n))

    psi = np.zeros(((m + 2) * (n + 2)), dtype=np.float32)
    psitmp = np.zeros(((m + 2) * (n + 2)), dtype=np.float32)
    i, j = 0, 0

    psi_buf = cl.Buffer(p.getContext(),p.getFlags().READ_WRITE | p.getFlags().COPY_HOST_PTR, hostbuf=psi)
    psitmp_buf = cl.Buffer(p.getContext(), p.getFlags().READ_WRITE | p.getFlags().COPY_HOST_PTR, hostbuf=psitmp)
#loop1
    p.getProgram().loop1(p.getQueue(), psi.shape, None, psi_buf, np.int32(b+1), np.int32(b+w), np.int32(m)).wait()
    cl._enqueue_read_buffer(p.getQueue(), psi_buf, psi).wait()
#loop2
    p.getProgram().loop2(p.getQueue(), psi.shape, None, psi_buf, np.int32(b + w), np.int32(m+1), np.int32(w)).wait()
    cl._enqueue_read_buffer(p.getQueue(), psi_buf, psi).wait()
#loop3
    p.getProgram().loop3(p.getQueue(), psi.shape, None, psi_buf, np.int32(h+1), np.int32(m), np.int32(w)).wait()
    cl._enqueue_read_buffer(p.getQueue(), psi_buf, psi).wait()
#loop4
    p.getProgram().loop4(p.getQueue(), psi.shape, None, psi_buf, np.int32(h+1), np.int32(h+w), np.int32(m), np.int32(w)).wait()
    cl._enqueue_read_buffer(p.getQueue(), psi_buf, psi).wait()

    bnorm = 0.0
    #loop5(zbrojiti listu bnorm)
    bnorm_list = np.zeros(((m + 3)), dtype=np.float32)
    bnorm_buf = cl.Buffer(p.getContext(), p.getFlags().READ_WRITE | p.getFlags().COPY_HOST_PTR, hostbuf=bnorm_list)
    p.getProgram().loop5(p.getQueue(), psi.shape, None, psi_buf, np.int32(m + 2), np.int32(n+2), bnorm_buf ).wait()
    cl._enqueue_read_buffer(p.getQueue(), bnorm_buf, bnorm_list).wait()

    bnorm=sum(bnorm_list)
    bnorm = math.sqrt(bnorm)

    print("\nStarting main loop...\n\n")
    tstart = time.time()

    for iter in range(1, numiter + 1):
        # loop6
        p.getProgram().loop6(p.getQueue(), psi.shape, None, psi_buf, psitmp_buf, np.int32(m + 1), np.int32(n + 1)).wait()
        cl._enqueue_read_buffer(p.getQueue(), psitmp_buf, psitmp).wait()

        if checkerr or iter == numiter:
            dsq = 0
            dsq_list = np.zeros(((m + 2)), dtype=np.float32)
            dsq_buf = cl.Buffer(p.getContext(), p.getFlags().READ_WRITE | p.getFlags().COPY_HOST_PTR,
                                  hostbuf=dsq_list)
            p.getProgram().loop7(p.getQueue(), psi.shape, None, psi_buf, psitmp_buf, np.int32(m + 1), np.int32(n + 1), dsq_buf).wait()
            cl._enqueue_read_buffer(p.getQueue(), dsq_buf, dsq_list).wait()
            dsq=sum(dsq_list)
            # loop7(zbroji dsq)
            error = dsq
            error = math.sqrt(error)
            error = error / bnorm

        if checkerr:
            if (error < tolerance):
                print("Converged on iteration {}\n".format(iter))
                break
        #loop8
        p.getProgram().loop8(p.getQueue(), psi.shape, None, psi_buf, psitmp_buf, np.int32(m + 1), np.int32(n + 1)).wait()
        cl._enqueue_read_buffer(p.getQueue(), psi_buf, psi).wait()


        if (iter % printfreq == 0):
            if not checkerr:
                print("Completed iteration {}\n".format(iter))
            else:
                print("Completed iteration {}, error = {}\n".format(iter, error))

    if iter > numiter:
        iter = numiter

    tstop = time.time()

    ttot = tstop - tstart
    titer = ttot / iter

    print("\n... finished\n")
    print("After {} iterations, the error is {}\n".format(iter, error))
    print("Time for {} iterations was {} seconds\n".format(iter, ttot))
    print("Each iteration took {} seconds\n".format(titer))
    print("... finished\n")


if __name__ == '__main__':
    main()