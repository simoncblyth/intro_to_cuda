//  nvcc -I.. TimTest.cu && ./a.out && rm a.out 

#include "NP.hh"
#include "Tim.hh"

/**
TimTest.cu
------------

**/

int main(int argc, char** argv)
{
    const char* dir = argc > 1 ? argv[1] : "/tmp/recon" ; 
    NP<double>* t = NP<double>::Load(dir, "t.npy") ;  
    if( t == NULL ) return 1 ; 

    Tim<double> tim(t);  
    tim.sums(); 
    tim.dump(); 

    return 0 ; 
}
