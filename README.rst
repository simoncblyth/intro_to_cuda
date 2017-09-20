Introduction To CUDA
=======================


CUDA Documentation/Download
-----------------------------

* http://docs.nvidia.com/cuda/index.html

Hello World Examples
----------------------

This repository contains a few very simple examples
of using CUDA and Thrust.

* https://bitbucket.org/simoncblyth/intro_to_cuda/src/

Start by cloning the repository to a machine with an NVIDIA GPU::

    which hg # Mercurial is required
    hg clone https://bitbucket.org/simoncblyth/intro_to_cuda

Below commands compile and run hello.cu::

    delta:~ blyth$ cd intro_to_cuda
    delta:intro_to_cuda blyth$ head -1 hello.cu  
    // nvcc hello.cu -run && rm a.out  
    delta:intro_to_cuda blyth$ which nvcc    ## check CUDA compiler is available in your PATH
    /Developer/NVIDIA/CUDA-7.0/bin/nvcc
    delta:intro_to_cuda blyth$ nvcc hello.cu -run && rm a.out 
    Hello World!
    Hello World (from mykernel)!
    delta:intro_to_cuda blyth$ 
     

GPU Intro
----------

* https://blogs.nvidia.com/blog/2009/12/16/whats-the-difference-between-a-cpu-and-a-gpu/


CUDA Introductions
--------------------

An Introduction to GPU Computing and CUDA Architecture, Sarah Tariq, NVIDIA 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://on-demand.gputechconf.com/gtc-express/2011/presentations/GTC_Express_Sarah_Tariq_June2011.pdf


Really Fast Introduction to CUDA and CUDA C, Dale Southard, NVIDIA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.nersc.gov/assets/Uploads/CUDAIntrosouthard.pdf




Thrust
----------

* http://on-demand.gputechconf.com/gtc/2012/presentations/S0602-Intro-to-Thrust-Parallel-Algorithms-Library.pdf

Some more advanced slides on Thrust:

* http://outreach.sbel.wisc.edu/Workshops/GPUworkshop/2012-polimi/presentation-day4.pdf


cudaMalloc : why void** ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    int* ptr = 0;
    void** ptr_to_ptr = &ptr;
    cudaMalloc(ptr_to_ptr, sizeof(int));
    assert(ptr != 0);
    // ptr now points to a segment of device memory



GTC Search for CUDA
------------------------

* http://on-demand-gtc.gputechconf.com/gtcnew/on-demand-gtc.php?searchByKeyword=CUDA&searchItems=&sessionTopic=&sessionEvent=&sessionYear=&sessionFormat=&submit=&select=




