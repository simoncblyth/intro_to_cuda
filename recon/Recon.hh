#pragma once

/**
Recon.hh interface header
=========================

This header exists to hide the CUDA implementation, enabling 
usage from code compiled by clang/gcc (ie not nvcc).

**/

#include <string>
#include <initializer_list>
#include <vector>

template <typename T> struct Rec ; 

template <typename T>
struct Recon
{
    Rec<T>* rec ; 
    T last ; 

    Recon( const char* dir ) ;  

    T nll( std::initializer_list<T> ini ); 
    T nll( const std::vector<T>& par ); 

    const std::vector<T>& get_param() const ; 
    const std::vector<std::string>& get_label() const ; 

    std::string desc() const ; 

};



