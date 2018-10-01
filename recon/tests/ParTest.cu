//  nvcc -I.. ParTest.cu -std=c++11 && ./a.out && rm a.out

#include "NP.hh"
#include "Par.hh"
#include <iomanip>

/**
ParTest.cu
------------

**/

int main(int argc, char** argv)
{
    const char* dir = argc > 1 ? argv[1] : "/tmp/recon" ; 
    NP<double>* p = NP<double>::Load(dir, "parTru.npy") ;  
    NP<unsigned char>* l = NP<unsigned char>::Load(dir, "parLab.npy") ;  
    if( p == NULL || l == NULL ) return 1 ; 

    Par<double> param(p, l);  
    param.dump(); 

    param.set( { 1., 2., 3. , 4. } ); 
    param.dump(); 

    param.save(dir, "parTest.npy") ; 
    // python3 -c "import numpy as np ; print(np.load('/tmp/recon/parTest.npy'))" 


    const std::vector<double>& par = param.get() ; 

    for( unsigned i=0 ; i < par.size() ; i++ ) 
         std::cout 
             << std::setw(2) << i 
             << " : " 
             << std::fixed << par[i]
             << std::endl 
             ;


    return 0 ; 
}
