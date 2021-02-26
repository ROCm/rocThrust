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

# GIT
find_package(Git REQUIRED)
if (NOT Git_FOUND)
  message(FATAL_ERROR "Please ensure Git is installed on the system")
endif()

if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
  # rocPRIM (https://github.com/ROCmSoftwarePlatform/rocPRIM)
  find_package(rocprim)
endif()
if(NOT rocprim_FOUND)
  message(STATUS "rocPRIM not found or force download rocPRIM on. Downloading and building rocprim.")
  set(ROCPRIM_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/rocprim CACHE PATH "")
  download_project(
    PROJ                rocprim
    GIT_REPOSITORY      https://github.com/ROCmSoftwarePlatform/rocPRIM.git
    GIT_TAG             develop
    INSTALL_DIR         ${ROCPRIM_ROOT}
    CMAKE_ARGS          -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_PREFIX_PATH=/opt/rocm
    LOG_DOWNLOAD        TRUE
    LOG_CONFIGURE       TRUE
    LOG_BUILD           TRUE
    LOG_INSTALL         TRUE
    BUILD_PROJECT       TRUE
    UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
  )
  find_package(rocprim REQUIRED CONFIG PATHS ${ROCPRIM_ROOT} NO_DEFAULT_PATH)
endif()

# Test dependencies
if(BUILD_TEST)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    # Google Test (https://github.com/google/googletest)
    find_package(GTest QUIET)
  endif()

  if(NOT GTEST_FOUND)
    message(STATUS "GTest not found or force download GTest on. Downloading and building GTest.")
    set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/gtest CACHE PATH "")
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
      UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
    )
  endif()
  find_package(GTest REQUIRED CONFIG PATHS ${GTEST_ROOT})

  if(LARGE_TEST)
    # rocRAND (https://github.com/ROCmSoftwarePlatform/rocRAND)
    if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
      find_package(rocrand QUIET)
    endif()

    if(NOT rocrand_FOUND)
      message(STATUS "rocRAND not found or force download rocRAND on. Downloading and building rocrand.")
      set(ROCRAND_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/rocrand CACHE PATH "")
      download_project(
        PROJ                rocrand
        GIT_REPOSITORY      https://github.com/ROCmSoftwarePlatform/rocRAND.git
        GIT_TAG             develop
        INSTALL_DIR         ${ROCRAND_ROOT}
        CMAKE_ARGS          -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_PREFIX_PATH=/opt/rocm
        LOG_DOWNLOAD        TRUE
        LOG_CONFIGURE       TRUE
        LOG_BUILD           TRUE
        LOG_INSTALL         TRUE
        BUILD_PROJECT       TRUE
        UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
      )
    endif()
    find_package(rocrand REQUIRED CONFIG PATHS ${ROCRAND_ROOT} NO_DEFAULT_PATH)
  endif()
endif()
