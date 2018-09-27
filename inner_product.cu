// nvcc inner_product.cu && ./a.out && rm a.out 

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>

//  http://thrust.github.io/doc/group__transformed__reductions.html
//  http://thrust.github.io/doc/functional_8h_source.html

int main()
{
    const int N = 1000;

    thrust::device_vector<float> V1(N);
    thrust::device_vector<float> V2(N);
   
    thrust::sequence(V1.begin(), V1.end(), 1);
    thrust::sequence(V2.begin(), V2.end(), 0);
    //thrust::fill(V2.begin(), V2.end(), 75);

    thrust::plus<float>  bop_plus ;
    thrust::minus<float> bop_minus ; 
    thrust::maximum<float> bop_maximum ; 

    float sod = thrust::inner_product(V1.begin(), V1.end(), V2.begin(), 0.f , bop_plus,   bop_minus );
    float mod = thrust::inner_product(V1.begin(), V1.end(), V2.begin(), 0.f , bop_maximum, bop_minus );

    std::cout << " sum-of-differences " << sod << std::endl ; 
    std::cout << " max-of-differences " << mod << std::endl ; 

    return 0 ; 
}
