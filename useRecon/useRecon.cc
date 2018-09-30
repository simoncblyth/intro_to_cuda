
#include "Recon.hh"
#include <iostream>

int main(int argc, char** argv)
{
    typedef double FP ;  // type must match that of the .npy files

    const char* dir = argc > 1 ? argv[1] : "/tmp/recon" ; 

    Recon<FP> recon(dir) ; 

    FP nll = recon.nll(0,0,0,1); 

    std::cout << "nll " << nll << std::endl ; 

    return 0 ; 
}

