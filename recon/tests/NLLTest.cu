// nvcc -I.. NLLTest.cu -std=c++11 && ./a.out && rm a.out 

#include <thrust/device_vector.h>
#include "NLL.hh"

int main(int argc, char** argv)
{
    int tnum = 1 ; 
    thrust::device_vector<double> t(tnum) ; 
    thrust::device_vector<double> x(tnum) ; 
    thrust::device_vector<double> y(tnum) ; 
    thrust::device_vector<double> z(tnum) ; 

    // NB this is very slow way to copy to GPU : just for low stats testing 
    x[0] = 0. ; 
    y[0] = 0. ; 
    z[0] = 20. ; 
    t[0] = 20. ;

    thrust::device_vector<double> p(4) ; 
    p[0] = 0. ; 
    p[1] = 0. ; 
    p[2] = 0. ; 
    p[3] = 1. ; 

              
    double* raw_t = thrust::raw_pointer_cast(t.data()) ;  
    double* raw_x = thrust::raw_pointer_cast(x.data()) ;  
    double* raw_y = thrust::raw_pointer_cast(y.data()) ;  
    double* raw_z = thrust::raw_pointer_cast(z.data()) ;  
    double* raw_p = thrust::raw_pointer_cast(p.data()) ;  


    NLL<double>* nll = new NLL<double>(raw_x, raw_y, raw_z, raw_t, raw_p ) ;  
         
    double v = -thrust::transform_reduce( 
              thrust::make_counting_iterator(0), 
              thrust::make_counting_iterator(tnum),
              *nll, 
              double(0),   
              thrust::plus<double>()) ; 

    //  python3 -c "import scipy.stats as ss ; print(-ss.norm.logpdf(20,20,1))"  0.9189385332046727
    std::cout << " v " << v << std::endl ;  

    return 0 ; 
}

