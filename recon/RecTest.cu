// nvcc -I$HOME/np RecTest.cu && ./a.out && rm a.out 

#include "Rec.hh"

int main(int argc, char** argv)
{
    typedef double FP ;  // type must match that of the .npy files

    const char* dir = argc > 1 ? argv[1] : "/tmp/recon" ; 

    Rec<FP> rec(dir) ;  
    rec.sums(); 

    FP nll = rec.nll_();  
    std::cout << "nll " << nll << std::endl ; 

    rec.par->set_param( 0,0,0,1 );  
    std::cout << "nll(0,0,0,1) " << rec.nll_() << std::endl ; 


    return 0 ; 
}

