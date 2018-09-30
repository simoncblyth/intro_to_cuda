cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(name ReconTest)

project(${name} VERSION 0.1.0 LANGUAGES CXX )  

set(TEST_CU
    TimTest.cu
    GeoTest.cu
    ParTest.cu
    RecTest.cu
)

set(TEST_CC
    ReconTest.cc
)

foreach(TEST_CU_SRC ${TEST_CU})
    get_filename_component(TGT ${TEST_CU_SRC} NAME_WE)
    cuda_add_executable(${TGT} ${TEST_CU_SRC})

    target_link_libraries(${TGT} Recon) 
    set_target_properties(${TGT} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


foreach(TEST_CC_SRC ${TEST_CC})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)
    add_executable(${TGT} ${TEST_CC_SRC})

    target_link_libraries(${TGT} Recon) 
    set_target_properties(${TGT} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

    install(TARGETS ${TGT} DESTINATION lib)
endforeach()





