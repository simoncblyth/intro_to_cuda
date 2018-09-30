
set(IMinuit2_MODULE "${CMAKE_CURRENT_LIST_FILE}")
set(IMinuit2_PREFIX "${CMAKE_INSTALL_PREFIX}/externals")

find_path(IMinuit2_INCLUDE_DIR 
   NAMES "Minuit2/FCNBase.h"
   PATHS 
        ${IMinuit2_PREFIX}/include
)

find_library(IMinuit2_LIBRARY
   NAMES Minuit2 
   PATHS
        ${IMinuit2_PREFIX}/lib
)

if(IMinuit2_INCLUDE_DIR AND IMinuit2_LIBRARY)
   set(IMinuit2_FOUND "YES")
else()
   set(IMinuit2_FOUND "NO")
endif()

if(IMinuit2_VERBOSE)
   message(STATUS "FindIMinuit2.cmake IMinuit2_MODULE      : ${IMinuit2_MODULE}  ")
   message(STATUS "FindIMinuit2.cmake IMinuit2_INCLUDE_DIR : ${IMinuit2_INCLUDE_DIR}  ")
   message(STATUS "FindIMinuit2.cmake IMinuit2_LIBRARY     : ${IMinuit2_LIBRARY}  ")
   message(STATUS "FindIMinuit2.cmake IMinuit2_FOUND       : ${IMinuit2_FOUND}  ")
endif()

set(tgt IntroToCUDA::IMinuit2)
if(IMinuit2_FOUND AND NOT TARGET ${tgt})
    add_library(${tgt} UNKNOWN IMPORTED) 
    set_target_properties(${tgt} 
         PROPERTIES 
            IMPORTED_LOCATION             "${IMinuit2_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${IMinuit2_INCLUDE_DIR}"
    )   
endif()



