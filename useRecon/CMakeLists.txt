cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name useRecon)
project(${name} VERSION 0.1.0 LANGUAGES CXX )  
include(IntroBuildOptions)

find_package(Recon REQUIRED CONFIG)

add_executable(${name} ${name}.cc)
target_link_libraries(${name}  IntroToCUDA::Recon)

install(TARGETS ${name} DESTINATION lib)

