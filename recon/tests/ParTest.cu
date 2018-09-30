//  nvcc -I.. ParTest.cu && ./a.out && rm a.out 

#include "NP.hh"
#include "Par.hh"

/**
ParTest.cu
------------

**/

int main(int argc, char** argv)
{
    typedef double FP ;  // type must match that of the .npy file

    const char* path = argc > 1 ? argv[1] : "/tmp/recon/parTru.npy" ; 
    NP<FP>* p = NP<FP>::Load(path) ;  
    if( p == NULL ) return 1 ; 

    Par<FP> par(p);  
    par.dump(); 

    par.set_param( 1., 2., 3. , 4. ); 
    par.dump(); 

    return 0 ; 
}
