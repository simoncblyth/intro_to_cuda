//  nvcc -I$HOME/np upload_to_device_vector_with_host_vector_strided.cu && ./a.out && rm a.out 

#include <numeric>
#include <iostream>

#include "NP.hh"
#include "recon/strided_range.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>


/**
upload_to_device_vector_with_host_vector_strided.cu
--------------------------------------------------------

Variation of upload_to_device_vector_with_host_vector.cu 
using a strided range interator to split the xyz into separate x,y,z 

1. NP load from file into thrust::host_vector 
2. sum all xyz values on host
3. used strided ranges to split the xyz into separate device x,y,z vectors  
4. add up all x, y and then z values on device

See also recon/Geo.hh which adapts this "demo" code 
into a struct for reusablility.

**/

int main(int argc, char** argv)
{
    typedef double FP ;  // type must match that of the .npy file

    const char* path = argc > 1 ? argv[1] : "/tmp/recon/sph.npy" ; 

    NP<FP>* xyz = NP<FP>::Load(path) ;  

    if( xyz == NULL ) return 1 ; 

    typedef thrust::host_vector<FP>::iterator Iterator;
  
    thrust::host_vector<FP> h_xyz(xyz->data.begin(), xyz->data.end() );  
    FP h_sum = thrust::reduce( h_xyz.begin(), h_xyz.end() ) ; 

    strided_range<Iterator> h_x(h_xyz.begin() + 0, h_xyz.end(), 3 );
    strided_range<Iterator> h_y(h_xyz.begin() + 1, h_xyz.end(), 3 );
    strided_range<Iterator> h_z(h_xyz.begin() + 2, h_xyz.end(), 3 );

    thrust::device_vector<FP> d_x(h_x.begin(), h_x.end()) ; 
    thrust::device_vector<FP> d_y(h_y.begin(), h_y.end()) ; 
    thrust::device_vector<FP> d_z(h_z.begin(), h_z.end()) ; 

    FP d_sum = 0 ; 
    d_sum = thrust::reduce( d_x.begin(), d_x.end(), d_sum );   
    d_sum = thrust::reduce( d_y.begin(), d_y.end(), d_sum );   
    d_sum = thrust::reduce( d_z.begin(), d_z.end(), d_sum );   

    std::cout << " h_sum " << h_sum << std::endl ;  
    std::cout << " d_sum " << d_sum << std::endl ;  

    return 0 ; 
}
