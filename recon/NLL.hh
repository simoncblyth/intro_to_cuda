#pragma once
#include <math_constants.h>


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
    T normlogpdf( T ob, T mu, T sg) 
    {
        // functional form verified in ~/intro_to_numpy/normal.py to match scipy.stats.norm.logpdf 
        return -(ob-mu)*(ob-mu)*0.5/(sg*sg) - 0.5*logf(2.*CUDART_PI*sg*sg) ;         
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

