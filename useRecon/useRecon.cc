
#include "Recon.hh"
#include <iostream>

int main(int argc, char** argv)
{
    const char* dir = argc > 1 ? argv[1] : "/tmp/recon" ; 

    Recon<double> recon(dir) ; 

    double nll = recon.nll( {0,0,0,1}); 

    std::cout << recon.desc() << " " << nll << std::endl ; 

    return 0 ; 
}

