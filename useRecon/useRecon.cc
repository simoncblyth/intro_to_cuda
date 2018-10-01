/**
Normally built using CMake machinery can be built from commandline::

   itc-
   clang -I$(itc-prefix)/include -std=c++11 -L$(itc-prefix)/lib -lRecon -lc++ -Wl,-rpath $(itc-prefix)/lib useRecon.cc && ./a.out && rm a.out

**/

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

