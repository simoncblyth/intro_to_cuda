// TEST=thrust_curand_estimate_pi om-t
// http://docs.nvidia.com/cuda/curand/device-api-overview.html#thrust-and-curand-example

#include <curand_kernel.h> 

struct estimate_pi 
{ 
    estimate_pi(int _N) : N(_N) {}

    __device__ float operator()(unsigned seed) 
    { 
        float sum = 0; 
        curandState rng; 
        curand_init(seed, 0, 0, &rng); 

        for(int i = 0; i < N; ++i) 
        { 
            float x = curand_uniform(&rng); 
            float y = curand_uniform(&rng); 
            float dist = sqrtf(x*x + y*y); 
            if(dist <= 1.0f) sum += 1.0f; 
        } 
        sum *= 4.0f; 
        return sum / N; 
    } 

    int N ; 
}; 



//#include <cmath>
#include <thrust/iterator/counting_iterator.h> 
//#include <thrust/functional.h> 
#include <thrust/transform_reduce.h> 
#include <iostream> 
#include <iomanip> 

int main(void) 
{ 
     int N = 10000; 
     int M = 30000; 

     float estimate = thrust::transform_reduce( 
                thrust::counting_iterator<int>(0), 
                thrust::counting_iterator<int>(M), 
                estimate_pi(N), 
                0.0f, 
                thrust::plus<float>()); 

      estimate /= M; 

      std::cout 
          << " M " << M 
          << " N " << N 
          << std::setprecision(5) 
          << std::fixed 
          << " estimate " 
          << estimate
          << " delta " 
          << estimate - M_PI
          << std::endl 
          ; 
      return 0; 
} 

