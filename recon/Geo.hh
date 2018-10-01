#include "strided_range.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

/**
Geo.hh
--------

Converts array-of-structs xyz coordinates into 
structure-of-arrays x,y,z as it uploads to the GPU using Thrust, 
the result is separate GPU buffers for x, y and z 
which is the recommended form to allow fast coalesced 
simultaneous memory access across parallel threads.

**/

template<typename T>
struct GeoDump
{
    T* x ; 
    T* y ; 
    T* z ; 

    GeoDump( T* x_, T* y_, T* z_ ) : x(x_), y(y_), z(z_) {}  

    __device__
    void operator()(int i)
    {
        T xi = x[i] ; 
        T yi = y[i] ; 
        T zi = z[i] ; 
        printf("GeoDump %i (%f %f %f) \n", i, xi, yi, zi );
    }
};


template <typename T>
struct Geo
{
    typedef typename thrust::host_vector<T>::iterator Iterator;

    NP<T>* xyz ; 

    thrust::host_vector<T> h_xyz ;  

    strided_range<Iterator> h_x ;
    strided_range<Iterator> h_y ;
    strided_range<Iterator> h_z ;

    thrust::device_vector<T> d_x ; 
    thrust::device_vector<T> d_y ; 
    thrust::device_vector<T> d_z ; 

    T* raw_x ; 
    T* raw_y ; 
    T* raw_z ; 

    Geo(NP<T>* xyz_);
    void sums(); 
    void dump(); 
};


template <typename T>
Geo<T>::Geo(NP<T>* xyz_)
    :
    xyz(xyz_),
    h_xyz(xyz->data.begin(), xyz->data.end() ),

    h_x(h_xyz.begin() + 0, h_xyz.end(), 3 ),
    h_y(h_xyz.begin() + 1, h_xyz.end(), 3 ),
    h_z(h_xyz.begin() + 2, h_xyz.end(), 3 ),

    d_x(h_x.begin(), h_x.end()),
    d_y(h_y.begin(), h_y.end()),
    d_z(h_z.begin(), h_z.end()),

    raw_x(thrust::raw_pointer_cast(d_x.data())),
    raw_y(thrust::raw_pointer_cast(d_y.data())),
    raw_z(thrust::raw_pointer_cast(d_z.data()))
{
}

template <typename T>
void Geo<T>::sums()
{
    T h_sum = thrust::reduce( h_xyz.begin(), h_xyz.end() ) ; 
    std::cout << " h_sum " << h_sum << std::endl ;  

    T d_sum = 0 ; 
    d_sum = thrust::reduce( d_x.begin(), d_x.end(), d_sum );   
    d_sum = thrust::reduce( d_y.begin(), d_y.end(), d_sum );   
    d_sum = thrust::reduce( d_z.begin(), d_z.end(), d_sum );   

    std::cout << " d_sum " << d_sum << std::endl ;  
}


template <typename T>
void Geo<T>::dump() 
{
    GeoDump<T> func(raw_x, raw_y, raw_z) ; 
    thrust::for_each( 
          thrust::make_counting_iterator(0), 
          thrust::make_counting_iterator(xyz->num_values()/3),
          func 
      ) ; 
}


