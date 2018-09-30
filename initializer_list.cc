// nvcc initializer_list.cu -std=c++11 && ./a.out && rm a.out 
// clang initializer_list.cc -std=c++11 -lc++ && ./a.out && rm a.out 

#include <iostream>
#include <vector>
#include <initializer_list>


template<typename T>
struct func
{
    void va( std::initializer_list<T> ini )
    {
         std::cout << __PRETTY_FUNCTION__ <<std::endl ;
         std::vector<T> vec(ini) ; 
         std::cout << " vec.size() " << vec.size() << std::endl ; 
         for(unsigned i=0 ; i < vec.size() ; i++) std::cout << vec[i] << std::endl ; 
    }
};


int main()
{
    func<int> i ; 
    i.va( {1,2,3} ) ; 

    func<unsigned> u ; 
    u.va( {1,2,3} ) ; 
 
    func<float> f ; 
    f.va( {1.1f,2.1f,3.1f} ) ;
 
    func<double> d ; 
    d.va( {1.2,2.2,3.2} ) ; 

    return 0 ; 
}
