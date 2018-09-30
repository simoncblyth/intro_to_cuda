#include "Recon.hh"
#include "Rec.hh"

template<typename T>
Recon<T>::Recon( const char* dir )
    : 
    rec(new Rec<T>(dir))
{
}

template<typename T>
T Recon<T>::nll(T px, T py, T pz, T sg )
{
    rec->set_param( px, py, pz, sg ); 
    return rec->nll_() ; 
}

// explicit template instanciation

template struct Recon<double> ;
template struct Recon<float> ;


