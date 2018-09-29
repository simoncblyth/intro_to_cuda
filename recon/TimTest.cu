//  nvcc -I$HOME/np TimTest.cu && ./a.out && rm a.out 

#include "NP.hh"
#include "Tim.hh"

/**
TimTest.cu
------------

**/

int main(int argc, char** argv)
{
    typedef double FP ;  // type must match that of the .npy file

    const char* path = argc > 1 ? argv[1] : "/tmp/recon/t.npy" ; 
    NP<FP>* t = NP<FP>::Load(path) ;  
    if( t == NULL ) return 1 ; 

    Tim<FP> tim(t);  
    tim.sums(); 
    tim.dump(); 

    return 0 ; 
}
