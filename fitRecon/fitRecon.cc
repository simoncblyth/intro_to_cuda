/**
Normally built using CMake generated Makefile, but can also build and run with commandline::

    itc- ; iminuit2- ; \
            clang -std=c++11 -lc++ \
           -I$(itc-prefix)/include \
           -I$(iminuit2-prefix)/include \
           -L$(itc-prefix)/lib -lRecon \
           -L$(iminuit2-prefix)/lib -lMinuit2 \
           -Wl,-rpath $(itc-prefix)/lib \
           fitRecon.cc && ./a.out && rm a.out

On Linux replace the clang line with "gcc -std=c++11 -lstdc++" 

**/

#include <iostream>
#include "Recon.hh"

#include "Minuit2/FCNBase.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"

namespace ROOT {
namespace Minuit2 {

struct FCN : public FCNBase
{
    Recon<double>* recon ; 

    FCN(Recon<double>* recon_) : recon(recon_) {}  

    double operator()(const std::vector<double>& par) const 
    {
        return recon->nll(par); 
    }
    double Up() const { return 0.5 ; }
};

}
}

using namespace ROOT::Minuit2;

int main(int argc, char** argv)
{
    const char* dir = argc > 1 ? argv[1] : "/tmp/recon" ; 

    Recon<double>* recon = new Recon<double>(dir) ; 

    FCN fcn(recon) ; 

    MnUserParameters upar;

    const std::vector<double>& par = recon->get_param(); 
    const std::vector<std::string>& lab = recon->get_label(); 
    assert( par.size() == lab.size() ) ;  

    for( unsigned i=0 ; i < par.size() ; i++)  upar.Add( lab[i].c_str(), par[i], 0.1 ) ;   // TODO: pass the err from recon  
   

    MnMigrad migrad(fcn, upar);

    FunctionMinimum min = migrad();

    std::cout<<"min= "<<min<<std::endl;

    return 0 ; 
}

