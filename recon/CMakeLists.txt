cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name Recon)

project(${name} VERSION 0.1.0 LANGUAGES CXX )  

include(IntroBuildOptions)

find_package(CUDA REQUIRED MODULE)

set(SOURCES Recon.cu)
set(HEADERS Recon.hh)

CUDA_ADD_LIBRARY(${name} SHARED ${SOURCES})
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )

#message(STATUS "CUDA_LIBRARIES:${CUDA_LIBRARIES}") 
#message(STATUS "CMAKE_INSTALL_LIBDIR:${CMAKE_INSTALL_LIBDIR}") 

install(TARGETS ${name}     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${HEADERS}            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE IntroToCUDA:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)

