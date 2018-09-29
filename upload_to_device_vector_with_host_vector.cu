//  nvcc -I$HOME/np upload_to_device_vector_with_host_vector.cu && ./a.out && rm a.out 

#include <numeric>
#include <iostream>

#include "NP.hh"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

/**
upload_to_device_vector_with_host_vector.cu
---------------------------------------------

Variation of upload_to_device_vector.cu that avoids the cudaMemcpy 


**/

int main(int argc, char** argv)
{
    typedef double FP ;  // type must match that of the .npy file

    const char* path = argc > 1 ? argv[1] : "/tmp/recon/sph.npy" ; 

    NP<FP>* sph = NP<FP>::Load(path) ;  

    if( sph == NULL ) return 1 ; 
   
    FP a_result = std::accumulate( sph->data.begin(), sph->data.end(), FP(0) ) ;   // sum all values on CPU
  
    std::cout << " a_result " << a_result << std::endl ; 

    thrust::host_vector<FP> h_sph(sph->data.begin(), sph->data.end() );  

    FP h_result = thrust::reduce( h_sph.begin(), h_sph.end() ) ;  

    std::cout << " h_result " << h_result << std::endl ; 

    thrust::device_vector<FP> d_sph(h_sph) ;   // GPU allocation 

    FP d_result = thrust::reduce( d_sph.begin(), d_sph.end() ) ;  // sum all values on GPU 

    std::cout 
         << " path " << path 
         << " a_result " << a_result 
         << " h_result " << h_result 
         << " d_result " << d_result 
         << " delta " << (d_result - h_result )
         << std::endl
         << "python -c \"import sys, numpy as np ; print(np.load(sys.argv[1]).sum()) \" " << path   
         << std::endl ; 


    return 0 ; 
}
