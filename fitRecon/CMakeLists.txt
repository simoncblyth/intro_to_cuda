cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name fitRecon)
project(${name} VERSION 0.1.0 LANGUAGES CXX )  
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#set(IMinuit2_VERBOSE ON)

find_package(Recon REQUIRED CONFIG)
find_package(IMinuit2 REQUIRED MODULE)

add_executable(${name} ${name}.cc)

target_link_libraries(${name} 
   IntroToCUDA::Recon
   IntroToCUDA::IMinuit2
)

install(TARGETS ${name} DESTINATION lib)

