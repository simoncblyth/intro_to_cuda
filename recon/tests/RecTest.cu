// nvcc -I.. RecTest.cu -std=c++11 && ./a.out && rm a.out 

#include "Rec.hh"

int main(int argc, char** argv)
{
    const char* dir = argc > 1 ? argv[1] : "/tmp/recon" ; 

    Rec<double> rec(dir) ;  
    rec.sums(); 

    std::cout << rec.desc() << " : " << rec.nll_() << std::endl ; 

    rec.set_param( {0,0,0,1} );  

    std::cout << rec.desc() << " : " << rec.nll_() << std::endl ; 

    rec.set_param( {0,0,-5,1} );  

    std::cout << rec.desc() << " : " << rec.nll_() << std::endl ; 


    const std::vector<double>& par = rec.get_param(); 
    const std::vector<std::string>& lab = rec.get_label();

    assert( par.size() == lab.size() ) ; 

    for( unsigned i=0 ; i < par.size() ; i++) 
        std::cout << lab[i] << " : " << par[i] << std::endl ;  


    return 0 ; 
}

