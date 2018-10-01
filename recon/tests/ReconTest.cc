/**

itc-
clang -I$(itc-prefix)/include -std=c++11 -L$(itc-prefix)/lib -lRecon -lc++ -Wl,-rpath $(itc-prefix)/lib ReconTest.cc && ./a.out && rm a.out

**/

#include <initializer_list>
#include <iostream>
#include "Recon.hh"

int main(int argc, char** argv)
{
    const char* dir = argc > 1 ? argv[1] : "/tmp/recon" ; 

    Recon<double> recon(dir) ; 

    for( double z=-10. ; z< 11. ; z+= 1. )
    { 
        double nll = recon.nll( {0,0,z,1.} ); 
        std::cout << recon.desc() << std::endl ; 
    }

    return 0 ; 
}

