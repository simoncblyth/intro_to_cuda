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
        double x = par[0];
        double y = par[1];
        double z = par[2];
        double s = par[3];
        return recon->nll(x,y,z,s); 
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

    upar.Add("x", 0., 0.1); 
    upar.Add("y", 0., 0.1); 
    upar.Add("z", 0., 0.1); 
    upar.Add("s", 1., 0.1); 

    MnMigrad migrad(fcn, upar);

    FunctionMinimum min = migrad();

    std::cout<<"min= "<<min<<std::endl;

    return 0 ; 
}

