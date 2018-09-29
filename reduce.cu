// nvcc reduce.cu && ./a.out && rm a.out 

#include <cassert>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

int main()
{
    int N = 10 ; 

    thrust::device_vector<float> d_vec(N) ; 

    thrust::sequence( d_vec.begin(), d_vec.end() ) ; 

    float sum = thrust::reduce( d_vec.begin(), d_vec.end() ) ; 

    float x_sum = 0.f ; 
    for( int i=0 ; i < N ; i++ ) x_sum += float(i) ; 

    std::cout << " sum " << sum << std::endl ; 

    assert( sum == x_sum ) ; 


    return 0 ; 

}
