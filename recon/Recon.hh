#pragma once

/**
Recon.hh interface header
=========================

This header exists to hide the CUDA implementation, enabling 
usage from code compiled by clang/gcc.

**/

template <typename T> struct Rec ; 

template <typename T>
struct Recon
{
    Rec<T>* rec ; 
    Recon( const char* dir ) ;  
    T nll(T px, T py, T pz, T sg ); 

};



