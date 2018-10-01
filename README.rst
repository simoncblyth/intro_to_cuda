Introduction To CUDA
=======================

CUDA Documentation/Download
-----------------------------

* http://docs.nvidia.com/cuda/index.html

See Also
----------

* https://bitbucket.org/simoncblyth/intro_to_numpy/

  NumPy and the NPY file format are exceedingly useful for transporting/persisting 
  GPU inputs and outputs, making learning NumPy go together naturally with learning CUDA and Thrust 


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
     

Brief descriptions of simple examples
-----------------------------------------------------

hello.cu
    simplest possible kernel 

add.cu
    pure CUDA addition

inner_product.cu
    shows how to use the generalized inner product thrust::inner_product to 
    get the maximum difference of two thrust device vectors 

printfTest.cu
    demonstrate printf from a thrust functor, note the cudaDeviceSynchronize 
    otherwise the process terminates before any output from kernel gets pumped
    to the terminal

thrust_sort.cu
    sorting 32M random integers on device 

thrust_curand_estimate_pi.cu
    use monte carlo method to estimate pi using thrust::tranform_reduce with 
    a functor with operator method that runs on device

texture/demoTex.cu
    simple sample code using a GPU texture from CUDA, Thrust is used 
    a little to avoid CUDA boilerplate code

texture/demoTex2.cu
    enhance the demoTex.cu example, putting the results of the texture lookups
    into a buffer and using that to compare texture lookup results with expectations.
    A generalized thrus::inner_product is used to find the maximum difference
    between results and expectations.

reduce.cu
    thrust::reduce adding all values in a sequence on GPU 

upload_to_device_vector.cu
    load values from .npy file using NP.hh from https://github.com/simoncblyth/np 
    cudaMemcpy from host to device, thrust::reduce on device
  
upload_to_device_vector_with_host_vector.cu
    varation using thrust::host_vector to avoid the cudaMemcpy and tricky pointer casting

upload_to_device_vector_with_host_vector_strided.cu
    use strided_range.h to to split xyz data on host into separate x,y,z on device :
    many sources suggest that this structure-of-array form performs better than
    array-of-structs due to more coalesced memory access between the parallel threads 
    

Extended Example : Minuit2 Fitting of GPU evaluated Log Likelihood
----------------------------------------------------------------------------------

The example is structured as several separate CMake configured projects.
Globally relevant files and directories are described here, for further details 
see the README.rst within the directories.

itc.bash
    top level control bash script

iminuit2.bash
    Minuit2 downloader and installer

ibcm.bash
    BCM (boost cmake modules) downloader and installer

cmake/Modules
    infrastructure : build options, flags and FindIMinuit2.cmake

recon
    code for the nvcc compiled and linked libRecon
    providind GPU NLL evaluation, implemented using CUDA Thrust.
    See recon/README.rst for details
    
useRecon
    clang/gcc compiled minimal usage of libRecon via Recon.hh

useIMinuit2
    minimal usage of Minuit2 external 

fitRecon
    brings together Minuit2 fitter with libRecon NLL on GPU  




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

  Including rainfall worked example, that uses struct-of-arrays (not array-of-structs), which get
  tied together using tuples and zip iterators.


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


Thrust API Documentation
--------------------------

* http://thrust.github.io
* http://thrust.github.io/doc/modules.html



Most Thrust Intros
--------------------

* http://on-demand.gputechconf.com/gtc/2010/presentations/S12219-High-Productivity-CUDA-Development-Thrust-Template-Library.pdf

  Includes a sorting float2 vertices example, for handling triangle soup 


* http://on-demand.gputechconf.com/gtc/2010/presentations/S12220-Thrust-By-Example-Advanced-Features-Techniques.pdf

  * Fusion using transform_iterator, avoiding intermediate result 
  * better to use transform_reduce rather than separate transform then reduce 
  * structure-of-arrays "soa" better coalesced memory access
  * zip_iterator and tuple gives conceptual goodness of array-of-structs "aos" but performance of struct-of-arrays "soa" 
  * 2d bucket sort example


* http://on-demand.gputechconf.com/supercomputing/2012/presentation/SB035-Bradley-Thrust-Parallel-Algorithms-Library.pdf





Advanced CUDA References
--------------------------

* http://on-demand.gputechconf.com/gtc/2013/presentations/S3049-Getting-Started-CUUA-C-PlusPlus.pdf

* http://on-demand.gputechconf.com/gtc/2010/presentations/S12084-State-of-Art-GPU-Data-Parallel-Algorithm-Primitives.pdf



Advanced Thrust References
---------------------------

* http://www.mariomulansky.de/data/uploads/cuda_thrust.pdf

  * make_transform_iterator
  * make_zip_iterator
  * make_tuple
  * for_each
  * Numerical Integration of an ODE, writing into a tuple from the functor
  * make_permutation_iterator

* https://www.nvidia.com/docs/IO/116711/sc11-montecarlo.pdf

  * estimate pi without using a functor, using thrust::count 

* http://on-demand.gputechconf.com/gtc/2015/presentation/S5338-Bharatkumar-Sharma.pdf

  Thrust++ using thrust in medical imaging 

* http://on-demand.gputechconf.com/gtc/2016/presentation/s6431-steven-dalton-advanced-thrust-programming.pdf

  Thrust execution policy 

* http://www.bu.edu/pasi/files/2011/07/Lecture6.pdf

  * covers iterators in depth
  * fusion using transform_reduce
  * rotate 3d vectors stored as struct-of-arrays using zip_iterator and tuples



GTC Search for CUDA
------------------------

* https://on-demand-gtc.gputechconf.com/gtcnew/on-demand-gtc.php?searchByKeyword=Thrust%20&searchItems=&sessionTopic=&sessionEvent=&sessionYear=&sessionFormat=&submit=&select=


Alternatives to Thrust 
-------------------------

* http://nvlabs.github.io/cub/

* https://moderngpu.github.io/intro.html






