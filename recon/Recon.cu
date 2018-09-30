
#include <sstream>
#include <initializer_list>

#include "Recon.hh"
#include "Rec.hh"

template<typename T>
Recon<T>::Recon( const char* dir )
    : 
    rec(new Rec<T>(dir)), 
    last(0)
{
}

template<typename T>
T Recon<T>::nll( std::initializer_list<T> ini )
{
    rec->set_param( ini ); 
    last = rec->nll_(); 
    return last ; 
}

template<typename T>
T Recon<T>::nll( const std::vector<T>& par )
{
    rec->set_param( par ); 
    last = rec->nll_(); 
    return last ; 
}

template<typename T>
std::string Recon<T>::desc() const 
{
    std::stringstream ss ; 
    ss << rec->desc() 
       << " : "
       << std::fixed << last 
       ;
    return ss.str(); 
}


template<typename T>
const std::vector<T>& Recon<T>::get_param() const 
{
    return rec->get_param(); 
}

template<typename T>
const std::vector<std::string>& Recon<T>::get_label() const 
{
    return rec->get_label(); 
}

// explicit template instanciation

template struct Recon<double> ;
template struct Recon<float> ;


