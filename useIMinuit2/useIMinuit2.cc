#include <iostream>

#include "Minuit2/FCNBase.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"

namespace ROOT {
namespace Minuit2 {

struct FCN : public FCNBase
{
    double operator()(const std::vector<double>& par) const {
        double x = par[0];
        return ( x*x );
    }
    double Up() const { return 0.5 ; }
};

}
}

using namespace ROOT::Minuit2;

int main(int argc, char** argv)
{
    FCN fcn ; 
    MnUserParameters upar;

    upar.Add("x", 1., 0.1); 

    MnMigrad migrad(fcn, upar);

    FunctionMinimum min = migrad();

    std::cout<<"min= "<<min<<std::endl;

    return 0 ; 
}
