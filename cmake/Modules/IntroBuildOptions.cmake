

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(COMPUTE_CAPABILITY 30)
include(IntroCUDAFlags)

include(GNUInstallDirs)
find_package(BCM)
include(BCMDeploy)
set(CMAKE_CXX_STANDARD 11)


