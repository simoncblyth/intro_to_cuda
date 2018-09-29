//  nvcc -I$HOME/np upload_to_device_vector.cu && ./a.out && rm a.out 

#include <numeric>
#include <iostream>

#include "NP.hh"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

/**
upload_to_device_vector.cu
---------------------------

1. loads npy array from file using NP.hh 
2. adds all values on CPU 
3. allocatates GPU memory using thrust::device_vector
4. cudaMemcpy the array to GPU
5. adds all values on GPU with thrust::reduce
6. compares the CPU and GPU totals

**/

int main(int argc, char** argv)
{
    typedef double FP ;  // type must match that of the .npy file

    const char* path = argc > 1 ? argv[1] : "/tmp/recon/sph.npy" ; 

    NP<FP>* sph = NP<FP>::Load(path) ;  

    if( sph == NULL ) return 1 ; 
   
    FP x_result = std::accumulate( sph->data.begin(), sph->data.end(), FP(0) ) ;   // sum all values on CPU
  
    std::cout << " x_result " << x_result << std::endl ; 

    thrust::device_vector<FP> d_sph(sph->num_values()) ;   // GPU allocation 

    FP* d_sph_ptr = thrust::raw_pointer_cast(d_sph.data());
   
    cudaMemcpy( (void*)d_sph_ptr, (const void*)sph->values(), sph->num_bytes(), cudaMemcpyHostToDevice );

    FP result = thrust::reduce( d_sph.begin(), d_sph.end() ) ;  // sum all values on GPU 

    std::cout 
         << " path " << path 
         << " result " << result 
         << " x_result " << x_result 
         << " delta " << (result - x_result )
         << std::endl
         << "python -c \"import sys, numpy as np ; print(np.load(sys.argv[1]).sum()) \" " << path   
         << std::endl ; 


    return 0 ; 
}
