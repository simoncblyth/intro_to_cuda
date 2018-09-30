cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name useIMinuit2)
project(${name} VERSION 0.1.0 LANGUAGES CXX )  
include(IntroBuildOptions)

set(IMinuit2_VERBOSE ON)
find_package(IMinuit2 REQUIRED MODULE)


add_executable(${name} ${name}.cc)
target_link_libraries(${name}  IntroToCUDA::IMinuit2)

install(TARGETS ${name} DESTINATION lib)

