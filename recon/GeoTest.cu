//  nvcc -I$HOME/np GeoTest.cu && ./a.out && rm a.out 

#include "NP.hh"
#include "Geo.hh"

/**
GeoTest.cu
------------

Developed from upload_to_device_vector_with_host_vector_strided.cu

1. loads xyz data from file
2. uploads to separate GPU buffers for x,y and z  

**/

int main(int argc, char** argv)
{
    typedef double FP ;  // type must match that of the .npy file

    const char* path = argc > 1 ? argv[1] : "/tmp/recon/sph.npy" ; 
    NP<FP>* xyz = NP<FP>::Load(path) ;  
    if( xyz == NULL ) return 1 ; 

    Geo<FP> geo(xyz);  
    geo.sums(); 
    geo.dump(); 

    return 0 ; 
}
