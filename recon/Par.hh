
#include <initializer_list>

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
    NP<unsigned char>* l ; 
    int npar ; 
    int nlab ; 
    std::vector<std::string> labs ; 

    thrust::host_vector<T> h_p ;  
    thrust::device_vector<T> d_p ; 
    T* raw_p ;  

    Par(NP<T>* p_, NP<unsigned char>* l_);

    void set( std::initializer_list<T> ini ) ; 
    void set( const std::vector<T>& par ) ; 
    const std::vector<T>& get() const ;  
    const std::vector<std::string>& labels() const ;  

    std::string desc() const ; 

    void sums(); 
    void dump(); 
};

template <typename T>
Par<T>::Par(NP<T>* p_, NP<unsigned char>* l_)
    :
    p(p_),
    l(l_),
    npar(p->num_values()),
    nlab(l->num_values()),
    h_p(p->data.begin(), p->data.end()),
    d_p(h_p),
    raw_p(thrust::raw_pointer_cast(d_p.data()))
{
    assert( npar == nlab ); 
    assert( npar == h_p.size() );  
    assert( npar == d_p.size() );  

    for( int i=0 ; i < nlab ; i++ ) 
    {
        char c[2] ; 
        c[0] = l->data[i] ;
        c[1] = '\0' ; 
        std::string s(c); 
        labs.push_back(s); 
    }
}

template <typename T>
void Par<T>::set(std::initializer_list<T> ini )
{
    std::vector<T> par(ini) ; 
    set(par); 
}

template <typename T>
void Par<T>::set(const std::vector<T>& par )
{
    // hmm : how to avoid so many copies of the parameters ?
    // perhaps can avoid h_p ??
    assert( par.size() == npar );  
    p->data.assign( par.begin(), par.end() ); 
    h_p.assign( par.begin(), par.end() ) ; 
    d_p = h_p ; 
}


template <typename T>
const std::vector<T>& Par<T>::get() const 
{
    return p->data ; 
}

template <typename T>
const std::vector<std::string>& Par<T>::labels() const 
{
    return labs ; 
}




template <typename T>
std::string Par<T>::desc() const 
{
    std::stringstream ss ; 
    ss << " { " ;
    for( int i=0 ; i < npar ; i++ ) 
        ss << labs[i] << ":" << std::fixed << h_p[i] << ( i == npar - 1 ? "" : ", " ) ; 
    ss << " } " ;
    return ss.str();  
}

template <typename T>
void Par<T>::dump()
{
    ParDump<T> func(raw_p) ; 
    thrust::for_each( 
          thrust::make_counting_iterator(0), 
          thrust::make_counting_iterator(npar),
          func 
      ) ; 
    std::cout << desc() << std::endl ; 
}


