# ########################################################################
# Copyright 2019 Advanced Micro Devices, Inc.
# ########################################################################

# ###########################
# rocThrust dependencies
# ###########################

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included.

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)
include(ExternalProject)
set(ROCRAND_URL "https://github.com/ROCmSoftwarePlatform/rocRAND.git" CACHE STRING "URL of git repository from which rocThrust will downloaded")
# GIT
find_package(Git REQUIRED)
if (NOT Git_FOUND)
  message(FATAL_ERROR "Please ensure Git is installed on the system")
endif()

# rocPRIM (https://github.com/ROCmSoftwarePlatform/rocPRIM)
if(NOT DOWNLOAD_ROCPRIM)
  find_package(rocprim)
endif()
if(NOT rocprim_FOUND)
  message(STATUS "Downloading and building rocprim.")
  download_project(
    PROJ                rocprim
    GIT_REPOSITORY      https://github.com/ROCmSoftwarePlatform/rocPRIM.git
    GIT_TAG             develop
    INSTALL_DIR         ${CMAKE_CURRENT_BINARY_DIR}/deps/rocprim
    CMAKE_ARGS          -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_PREFIX_PATH=/opt/rocm
    LOG_DOWNLOAD        TRUE
    LOG_CONFIGURE       TRUE
    LOG_BUILD           TRUE
    LOG_INSTALL         TRUE
    BUILD_PROJECT       TRUE
    UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
  )
  find_package(rocprim REQUIRED CONFIG PATHS ${CMAKE_CURRENT_BINARY_DIR}/deps/rocprim NO_DEFAULT_PATH)
endif()

# Test dependencies
if(BUILD_TEST)
  # Google Test (https://github.com/google/googletest)
  message(STATUS "Downloading and building GTest.")
  set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/gtest CACHE PATH "")
  download_project(
    PROJ                googletest
    GIT_REPOSITORY      https://github.com/google/googletest.git
    GIT_TAG             release-1.8.1
    INSTALL_DIR         ${GTEST_ROOT}
    CMAKE_ARGS          -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    LOG_DOWNLOAD        TRUE
    LOG_CONFIGURE       TRUE
    LOG_BUILD           TRUE
    LOG_INSTALL         TRUE
    BUILD_PROJECT       TRUE
    UPDATE_DISCONNECTED TRUE
  )
  find_package(GTest REQUIRED)
  find_package(rocrand QUIET)
  if(NOT rocrand_FOUND)
    message(STATUS " Downloading rocrand")
    ExternalProject_Add(rocrand
      PREFIX ${PROJECT_BINARY_DIR}
      GIT_REPOSITORY ${ROCRAND_URL}
      UPDATE_DISCONNECTED ${DISCONNECT}
      ${UPDATE_COMMAND_ARG}
      LIST_SEPARATOR |
      CMAKE_ARGS
        "-Wno-dev"
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DCMAKE_PREFIX_PATH=${PROJECT_BINARY_DIR}|${CMAKE_PREFIX_PATH_ALT_SEP}"
        "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
        "-DBUILD_TEST=OFF"
        "-DAMDGPU_TARGETS=${AMDGPU_TARGETS_ALT_SEP}"
    BUILD_ALWAYS ON
    )
    find_package(rocrand REQUIRED)
  endif()

endif()
