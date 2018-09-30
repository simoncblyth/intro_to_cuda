// nvcc useNaN.cu && ./a.out && rm a.out 

/**

epsilon:intro_to_cuda blyth$ nvcc useNaN.cu && ./a.out && rm a.out 
nan
nan
7ff8000000000000
 0 nan fff8000000000000 
  1 nan fff8000000000000 
  2 nan fff8000000000000 
  3 nan fff8000000000000 
  4 nan fff8000000000000 
  5 nan fff8000000000000 

**/

#include <iostream>
#include <cassert>
#include <cmath>
#include <math_constants.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h> 

union udl_t 
{
   double d ; 
   long   l ; 
};

struct Func
{
    __device__
    double operator()(int i)
    {
        double d = CUDART_NAN ; 

        udl_t udl ; 
        udl.d = d ; 

        printf( " %i %f %lx \n", i, d, udl.l ); 

        return d ;  
    }
};


int main()
{
    assert( sizeof(double) == sizeof(long) ); 
    Func func ; 

    thrust::for_each( thrust::make_counting_iterator(0), thrust::make_counting_iterator(10), func ) ;

    double d = nan("") ; 

    assert( isnan(d) == true ) ; 

    udl_t udl_cmath ; 
    udl_cmath.d = d ;

    std::cout << d << std::endl ; 
    std::cout << std::hex << d << std::endl ; 
    std::cout << std::hex << udl_cmath.l << std::endl ; 

    cudaDeviceSynchronize(); 

    return 0 ; 
}

