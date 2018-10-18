Recon 
=========

Overview
----------

This extended example demonstrates fitting a source position 
from times received at locations on a sphere using a simple
PDF model of a normal distribution around geometric time. 

The positions on the sphere, input times and initial parameter 
values and labels are loaded from files written by  

* https://bitbucket.org/simoncblyth/intro_to_numpy/src/default/recon.py 

To run `recon.py` you will need to install at least NumPy and preferably 
also SciPy, IPython and MatPlotLib as instructed at 

* https://bitbucket.org/simoncblyth/intro_to_numpy/src/default/


Installation and Building 
---------------------------

::

    cd
    hg clone http://bitbucket.org/simoncblyth/intro_to_cuda 

Add the following line to your .bash_profile or .bashrc::

    itc-(){  . $HOME/intro_to_cuda/itc.bash && itc-env ; }     

To install externals::

    itc-        # precursor function defines others such as itc-info, ibcm- iminuit2-

    ibcm-
    ibcm--      # downloads and installs BCM (Boost CMake Modules)

    iminuit2-
    iminuit2--  # downloads and installs Minuit2 


The above installs can now be done with one command::

    itc-
    itc-externals-install


To control the directory where the install is done set the *ITC_BASE* envvar::

    epsilon:recon blyth$ type itc-base       ## introspecting the function
    itc-base is a function
    itc-base () 
    { 
        echo ${ITC_BASE:-/tmp/$USER}
    }

For example::

    sudo mkdir /usr/local/intro_to_cuda
    sudo chown blyth:staff /usr/local/intro_to_cuda

The add a line to *~/.bash_profile*::

    export ITC_BASE=/usr/local


Build libRecon::

    cd ~/intro_to_cuda/recon
    ./go.sh   # configure, build and install libRecon using itc-cmake and itc-make

Minimal testing of libRecon::

    cd ~/intro_to_cuda/useRecon
    ./go.sh   # configure, build and run 

Fits times at sphere positions using libRecon and Minuit2::

    cd ~/intro_to_cuda/fitRecon
    ./go.sh   # configure, build and run 


Tips for developing code that uses CUDA Thrust
-------------------------------------------------

When developing code that uses Thrust minor errors can result in thousands of
lines of output from the nvcc compiler. This is because Thrust pushes what can be done 
with C++ templates. 

As a result the most efficient way to develop Thrust using code is to follow the
pattern of this example with small standalone headers with the entire implementation in the header 
that can be tested one by one. 

Implementation-in-the-header and building up functionality by adding more headers
works well with the nvcc compiler as it then gets to see the entire code of a "library" 
at once allowing more optimizations than possible when seeing individual
compilation units for linking together.


Infrastructure
----------------

CMakeLists.txt
    configuration 

go.sh
    building 

NP.hh NPU.hh
    loading/saving NPY formatted NumPy arrays

strided_range.h
    used by Geo.hh to extract x,y,z from xyz... 
 

Self contained headers and tests
----------------------------------

Tim.hh
    uploads times to GPU using Thrust     

Geo.hh
    converts array-of-structs xyz coordinates into structure-of-arrays 
    as it uploads to the GPU using Thrust, the result is separate 
    GPU buffers for x, y, z which is the recommended form to allow
    fast coalesced memory access across parallel threads  
    
Par.hh
    parameter upload and set/get interface with labels

NLL.hh
    GPU evaluation of negative likelihood of the PDF model
    using thrust::transform_reduce, the example PDF model is a normal 
    distrib around geometric time

Rec.hh
    brings together Geo,Tim,Par and NLL providing nll_() method

tests/NLLTest.cu
    totally standalone test without even loading files

tests/TimTest.cu tests/GeoTest.cu tests/ParTest.cu tests/RecTest.cu
    loads from file, uploads to GPU, does some on GPU dumping 


All these headers can be compiled and tested individually by the correspondingly 
named tests, see the first line for a comment with the commandline to build and run, eg::

     nvcc -I.. NLLTest.cu -std=c++11 && ./a.out && rm a.out 


Bridging between nvcc and gcc/clang code
--------------------------------------------

Recon.cu 
    includes Recon.hh and Rec.hh which brings in all the headers
    described above which have their entire implementations in the header.  
    This is compiled into libRecon.so or libRecon.dylib  

Recon.hh
    interface to libRecon functionality, the CUDA implementation
    is hidden behind the `Rec<T>` pointer.  Recon.hh can be compiled 
    by gcc/clang as well as nvcc allowing it to act as a bridge between the two.  

ReconTest.cc
    minimal use of libRecon : scanning the NLL  


Enhancement Ideas
-------------------

Provide a way to fit multiple events. 

* using higher dimensional NumPy time arrays ?
* implement way to update just the times, possibly with `thrust::copy`

 


