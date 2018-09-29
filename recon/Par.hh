
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>


template<typename T>
struct ParDump
{
    T* pp ; 

    ParDump(T* p_) : pp(p_) {}  


    __device__
    void operator()(int i)
    {
        T p = pp[i] ; 
        printf("ParDump %i %f \n", i, p );
    }
};


template <typename T>
struct Par
{
    typedef typename thrust::host_vector<T>::iterator Iterator;

    NP<T>* p ; 
    thrust::host_vector<T> h_p ;  
    thrust::device_vector<T> d_p ; 
    T* raw_p ;  

    Par(NP<T>* p_);

    void set_param( T p0, T p1, T p2, T p3) ; 
    void sums(); 
    void dump(); 
};

template <typename T>
Par<T>::Par(NP<T>* p_)
    :
    p(p_),
    h_p(p->data.begin(), p->data.end()),
    d_p(h_p),
    raw_p(thrust::raw_pointer_cast(d_p.data()))
{
}

template <typename T>
void Par<T>::set_param(T p0, T p1, T p2, T p3)
{
    d_p[0] = p0 ;
    d_p[1] = p1 ;
    d_p[2] = p2 ;
    d_p[3] = p3 ;
}

template <typename T>
void Par<T>::dump()
{
    ParDump<T> func(raw_p) ; 
    thrust::for_each( 
          thrust::make_counting_iterator(0), 
          thrust::make_counting_iterator(p->num_values()),
          func 
      ) ; 
}






