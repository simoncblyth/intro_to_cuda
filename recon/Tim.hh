#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>


template<typename T>
struct TimDump
{
    T* t ; 

    TimDump( T* t_ ) : t(t_) {}  

    __device__
    void operator()(int i)
    {
        T ti = t[i] ; 
        printf("TimDump %i %f \n", i, ti );
    }
};




template <typename T>
struct Tim
{
    typedef typename thrust::host_vector<T>::iterator Iterator;

    NP<T>* t ; 
    thrust::host_vector<T> h_t ;  
    thrust::device_vector<T> d_t ; 
    T* raw_t ;  

    Tim(NP<T>* t_);
    void sums(); 
    void dump(); 
};

template <typename T>
Tim<T>::Tim(NP<T>* t_)
    :
    t(t_),
    h_t(t->data.begin(), t->data.end()),
    d_t(h_t),
    raw_t(thrust::raw_pointer_cast(d_t.data()))
{
}

template <typename T>
void Tim<T>::sums()
{
    T h_sum = thrust::reduce( h_t.begin(), h_t.end() ) ; 
    std::cout << " h_sum " << h_sum << std::endl ;  

    T d_sum = thrust::reduce( d_t.begin(), d_t.end() );   
    std::cout << " d_sum " << d_sum << std::endl ;  
}


template <typename T>
void Tim<T>::dump()
{
    TimDump<T> func(raw_t) ; 
    thrust::for_each( 
          thrust::make_counting_iterator(0), 
          thrust::make_counting_iterator(t->num_values()),
          func 
      ) ; 
}






