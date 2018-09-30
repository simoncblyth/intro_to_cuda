#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <math_constants.h>

#include "NP.hh"
#include "Tim.hh"
#include "Geo.hh"
#include "Par.hh"

template<typename T>
struct NLL
{
    T* xx ; 
    T* yy ; 
    T* zz ; 
    T* tt ; 

    T* pp ;   

    NLL( T* x_, T* y_, T* z_, T* t_, T* p_ ) : xx(x_), yy(y_), zz(z_), tt(t_), pp(p_) {}  


    __device__
    T normlogpdf( T x, T mu, T sg) 
    {
        return -(x-mu)*(x-mu)*0.5/(sg*sg) - 0.5*logf(2.*CUDART_PI*sg*sg) ;         
    }

    __device__
    T operator()(int i)
    {
        T t = tt[i] ;

        T x = xx[i] ; 
        T y = yy[i] ; 
        T z = zz[i] ; 

        // assuming parTru.npy  has 4 param
        T px = pp[0] ; 
        T py = pp[1] ; 
        T pz = pp[2] ; 
        T psigma = pp[3] ;   

        T d = sqrt( (x-px)*(x-px) + (y-py)*(y-py) + (z-pz)*(z-pz) ) ;   
        T nlp = normlogpdf(t, d, psigma) ; 
 
        //printf("NLL %i (%f) (%f,%f,%f)  (%f,%f,%f,%f) d:%f nlp:%f \n", i, t, x,y,z, px,py,pz,psigma, d, nlp );

        return nlp  ;
    }
};

template<typename T>
struct Rec
{
    NP<T>* t ; 
    NP<T>* sph ; 
    NP<T>* p ; 

    Geo<T>* geo ;  
    Tim<T>* tim ;  
    Par<T>* par ;  
    NLL<T>* nll ; 

    Rec(const char* dir); 

    void set_param(T p0, T p1, T p2, T p3);
    void sums(); 
    T nll_();  
};

template<typename T>
Rec<T>::Rec(const char* dir)  
    :
    t(NP<T>::Load(dir, "t.npy")), 
    sph(NP<T>::Load(dir, "sph.npy")),
    p(NP<T>::Load(dir,"parTru.npy")),
    geo(new Geo<T>(sph)),
    tim(new Tim<T>(t)),
    par(new Par<T>(p)),
    nll(new NLL<T>(geo->raw_x, geo->raw_y, geo->raw_z, tim->raw_t, par->raw_p ))
{
    t->dump(0,10); 
    sph->dump(0,10); 
    assert( t->num_values() == sph->num_values()/3 ) ; 
} 

template <typename T>
void Rec<T>::set_param(T p0, T p1, T p2, T p3)
{
    par->set_param(p0,p1,p2,p3);
}

template<typename T>
void Rec<T>::sums()
{
    geo->sums();
    tim->sums();
}

template<typename T>
T Rec<T>::nll_()
{
    return -thrust::transform_reduce( 
              thrust::make_counting_iterator(0), 
              thrust::make_counting_iterator(t->num_values()),
              *nll, 
              T(0),   
              thrust::plus<T>()
           );
} 




