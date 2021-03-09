# ########################################################################
# Copyright 2019 Advanced Micro Devices, Inc.
# ########################################################################

# ###########################
# rocThrust dependencies
# ###########################

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included.

include_guard()

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

  # GTest has created a mess with the results of their FindModule and PackageConfig script results differing:
  #
  # MODULE: GTest::GTest, GTest::Main
  # CONFIG: GTest::gtest, GTest::gtest_main, GTest::gmock, GTest::gmock_main
  #
  # In the context of Dependencies.cmake, 4 scenarios may have happened:
  #
  #   1. CONFIG detection succeeds
  #   2. MODULE detection succeeds
  #   3. Neither succeed
  #   4. find_package wasn't even invoked
  #
  # We have to handle all 4 cases. Because we cannot ALIAS targets which are IMPORTED, we wrap them with a variable,
  # as ugly as it is.

  # 3. & 4. both require downloading and building GTest and is easiest to detect as the complement of 1. & 2.
  #
  # Note: because we're restricted to CMake 3.10 which doesn't have if(TARGET) yet, and testing for the existence
  #       of IMPORTED targets is otherwise, we check for side-effects of MODULE and CONFIG detection.
  if(EXISTS GTest_CONFIG)
    set(GTEST_CONFIG_SUCCEED True)
  else()
    set(GTEST_CONFIG_SUCCEED False)
  endif()
  if(DEFINED GTEST_FOUND)
    set(GTEST_MODULE_SUCCEED ${GTEST_FOUND})
  else()
    set(GTEST_MODULE_SUCCEED False)
  endif()

  if((NOT GTEST_MODULE_SUCCEED) AND (NOT GTEST_CONFIG_SUCCEED)) # 3. & 4.
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
    find_package(GTest REQUIRED CONFIG PATHS ${GTEST_ROOT})
    set(GTest_IMPORTED_targets GTest::gtest GTest::gtest_main)
  elseif(GTEST_CONFIG_SUCCEED) # 1.
    set(GTest_IMPORTED_targets GTest::gtest GTest::gtest_main)
  elseif(GTEST_MODULE_SUCCEED) # 2.
    set(GTest_IMPORTED_targets GTest::GTest GTest::Main)
  endif()

  if(LARGE_TEST)
    if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
      # rocRAND (https://github.com/ROCmSoftwarePlatform/rocRAND)
      find_package(rocrand QUIET)
      # TBB (https://github.com/oneapi-src/oneTBB)
      find_package(TBB QUIET)
    endif()

    if(NOT rocrand_FOUND)
      message(STATUS "rocRAND not found or force download rocRAND on. Downloading and building rocRAND.")
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

    if(NOT TBB_FOUND)
      message(STATUS "TBB not found or force download TBB on. Downloading and building TBB.")
      set(TBB_INSTALL_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/tbb CACHE PATH "")
      if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$")
        # hip-clang cannot compile googlebenchmark for some reason
        set(COMPILER_OVERRIDE "-DCMAKE_CXX_COMPILER=g++")
      endif()
      download_project(
        PROJ                tbb
        GIT_REPOSITORY      https://github.com/oneapi-src/oneTBB.git
        GIT_TAG             v2021.1.1
        INSTALL_DIR         ${TBB_INSTALL_ROOT}
        CMAKE_ARGS          -DTBB_TEST=OFF -DTBB_STRICT=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> ${COMPILER_OVERRIDE}
        LOG_DOWNLOAD        TRUE
        LOG_CONFIGURE       TRUE
        LOG_BUILD           TRUE
        LOG_INSTALL         TRUE
        BUILD_PROJECT       TRUE
        UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
      )
      unset(COMPILER_OVERRIDE)
    endif()
    find_package(TBB REQUIRED CONFIG PATHS ${TBB_INSTALL_ROOT} NO_DEFAULT_PATH)
  endif()
endif()
