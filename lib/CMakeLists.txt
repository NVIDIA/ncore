cmake_minimum_required(VERSION 3.10)
project(av_utils)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Download NumpyEigen if we haven't already
set(EXTERNAL_DEP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
include(DownloadExternalDeps)
download_dep(numpyeigen
  GIT_REPOSITORY https://github.com/fwilliams/numpyeigen.git
  GIT_TAG 7252ee0d8ae51414bbca3e0445ecfc285930b285
)
list(APPEND CMAKE_MODULE_PATH ${EXTERNAL_DEP_DIR}/numpyeigen/cmake)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(numpyeigen) # This will define Eigen3::Eigen if we enabled NPE_WITH_EIGEN

# Create the python module
npe_add_module(av_utils
  BINDING_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
  )

target_include_directories(av_utils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/nanoflann)
target_include_directories(av_utils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(av_utils PUBLIC OpenMP::OpenMP_CXX)
endif()