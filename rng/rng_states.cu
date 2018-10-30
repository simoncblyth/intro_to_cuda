// nvcc rng_states.cu && ./a.out && rm a.out && python -c "import numpy as np ; print(np.load('/tmp/rng.npy'))"

/**
rng_states.cu
===============

As curand_init requires a lot of stack, performance 
is drastically improved by splitting the curand initialization 
from its usage to generate random numbers.
This can be done by initializing, persisting the state, 
and then reloading this state once only into CUDA context
for use (by passing pointer to the buffer) by multiple 
other launches that need to generate randoms.

This approach ensures that the expensive initialization is only done once.

This code "rng_states.cu" is the starting point for exploring
how to do this in a more reusable fashion than is 
already implemented in Opticks. 
 

Opticks use of cuRAND : is not very reusable
------------------------------------------------

Opticks does separate CUDA launches to initialize cuRAND state:

https://bitbucket.org/simoncblyth/opticks/src/tip/cudarap/
https://bitbucket.org/simoncblyth/opticks/src/tip/cudarap/cuRANDWrapper.hh
https://bitbucket.org/simoncblyth/opticks/src/tip/cudarap/cuRANDWrapper.cc
https://bitbucket.org/simoncblyth/opticks/src/tip/cudarap/tests/cuRANDWrapperTest.cc

Loading the RNG state into OptiX context:

https://bitbucket.org/simoncblyth/opticks/src/tip/optixrap/ORng.cc
      
Other relevant Opticks code including some use of cuRAND from Thrust
(demo code, so it does the initialization every time) 
     https://bitbucket.org/simoncblyth/opticks/src/tip/thrustrap/tests/


**/

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include <curand_kernel.h>
#include "NP.hh"


__global__
void RNG_init( int seed, int offset, curandState* rng_states )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    curand_init(seed, id , offset, &rng_states[id]);
}

__global__
void RNG_gen( float* pa, curandState* ps )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    curandState rng = ps[id];  
    pa[id] = curand_uniform(&rng); 
    ps[id] = rng;     
}

struct RNG
{
    int seed ; 
    int offset ; 
    int num ; 
   
    thrust::host_vector<curandState>   hs ; 
    thrust::device_vector<curandState> ds ; 
    curandState*                       ps ;  

    NP<float>*                     na ;  
    thrust::host_vector<float>     ha ; 
    thrust::device_vector<float>   da ; 
    float*                         pa ; 

    RNG(int seed_, int num_) 
        : 
        seed(seed_),
        offset(0), 
        num(num_), 

        hs(num),
        ds(num),
        ps(thrust::raw_pointer_cast(ds.data())),

        na(new NP<float>(num)),
        ha(num),
        da(num),
        pa(thrust::raw_pointer_cast(da.data()))
    {
        init();
    }

    void init()
    {
        RNG_init<<<1,num>>>( seed, offset, ps ) ;  
    }

    void gen()
    {
        RNG_gen<<<1,num>>>( pa, ps ) ;  
        ha = da ; 
        std::copy( ha.begin(), ha.end(), na->data.begin() );  
    }
  
    void save(const char* path)
    {
        na->save(path); 
    }

};


int main()
{
     RNG rng(0, 10) ;      
     rng.gen();
     rng.save("/tmp/rng.npy"); 

     return 0 ; 
}



