#include <initializer_list>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include "NP.hh"
#include "Tim.hh"
#include "Geo.hh"
#include "Par.hh"
#include "NLL.hh"

template<typename T>
struct Rec
{
    const char* dir ; 
    NP<T>* t ; 
    NP<T>* s ; 
    NP<T>* p ; 
    NP<unsigned char>* l ; 

    int tnum ; 
    int snum ; 

    Geo<T>* geo ;  
    Tim<T>* tim ;  
    Par<T>* par ;  
    NLL<T>* nll ; 

    Rec(const char* dir_); 

    void set_param(std::initializer_list<T> ini);
    void set_param(const std::vector<T>& par_);
    const std::vector<T>& get_param() const ;
    const std::vector<std::string>& get_label() const ;

    T nll_() const ;  

    std::string desc() const ; 
    void save_param(const char* name="parFit.npy") ; 
    void sums(); 

};

template<typename T>
Rec<T>::Rec(const char* dir_)  
    :
    dir(strdup(dir_)),
    t(NP<T>::Load(dir, "t.npy")), 
    s(NP<T>::Load(dir, "sph.npy")),
    p(NP<T>::Load(dir, "parTru.npy")),
    l(NP<unsigned char>::Load(dir, "parLab.npy")),
    tnum(t->num_values()),
    snum(s->num_values()),
    geo(new Geo<T>(s)),
    tim(new Tim<T>(t)),
    par(new Par<T>(p,l)),
    nll(new NLL<T>(geo->raw_x, geo->raw_y, geo->raw_z, tim->raw_t, par->raw_p ))
{
    //t->dump(0,std::min(tnum,10)); 
    //s->dump(0,std::min(snum,10)); 
    assert( tnum == snum/3 ) ; 
} 

template <typename T>
void Rec<T>::set_param( std::initializer_list<T> ini)
{
    par->set(ini);
}
template <typename T>
void Rec<T>::set_param( const std::vector<T>& par_)
{
    par->set(par_);
}
template <typename T>
const std::vector<T>& Rec<T>::get_param() const
{
    return par->get();
}
template <typename T>
const std::vector<std::string>& Rec<T>::get_label() const
{
    return par->labels();
}

template<typename T>
T Rec<T>::nll_() const 
{
    return -thrust::transform_reduce( 
              thrust::make_counting_iterator(0), 
              thrust::make_counting_iterator(tnum),
              *nll, 
              T(0),   
              thrust::plus<T>()
           );
} 

template <typename T>
std::string Rec<T>::desc() const 
{
    std::stringstream ss ; 
    ss 
        << " tnum " << tnum 
        << " snum " << snum 
        << " par " << par->desc()
        ;
    return ss.str() ; 
}

template <typename T>
void Rec<T>::save_param(const char* name)
{
    par->save(dir, name);
}

template<typename T>
void Rec<T>::sums()
{
    geo->sums();
    tim->sums();
}

