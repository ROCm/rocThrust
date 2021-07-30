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
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    # Google Test (https://github.com/google/googletest)
    find_package(GTest QUIET)
  else()
    message(STATUS "Force installing GTest.")
  endif()

  if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)
    message(STATUS "GTest not found or force download GTest on. Downloading and building GTest.")
    set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/gtest CACHE PATH "")

    download_project(
      PROJ                googletest
      GIT_REPOSITORY      https://github.com/google/googletest.git
      GIT_TAG             release-1.10.0
      INSTALL_DIR         ${GTEST_ROOT}
      CMAKE_ARGS          -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE
    )
    list( APPEND CMAKE_PREFIX_PATH ${GTEST_ROOT} )
    find_package(GTest REQUIRED)
  endif()
endif()

if (WIN32)
    find_package(ROCM QUIET CONFIG PATHS /opt/rocm)
    if(NOT ROCM_FOUND)
      set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
      file(
        DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
        ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
        STATUS rocm_cmake_download_status LOG rocm_cmake_download_log
      )
      list(GET rocm_cmake_download_status 0 rocm_cmake_download_error_code)
      if(rocm_cmake_download_error_code)
        message(FATAL_ERROR "Error: downloading "
          "https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip failed "
          "error_code: ${rocm_cmake_download_error_code} "
          "log: ${rocm_cmake_download_log} "
        )
      endif()

      execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        RESULT_VARIABLE rocm_cmake_unpack_error_code
      )
      if(rocm_cmake_unpack_error_code)
        message(FATAL_ERROR "Error: unpacking  ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip failed")
      endif()
      find_package(ROCM REQUIRED CONFIG PATHS ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag})
    endif()
else()
  # Find or download/install rocm-cmake project
  find_package(ROCM 0.6 QUIET CONFIG PATHS /opt/rocm)
  if(NOT ROCM_FOUND)
    message(STATUS "rocm-cmake not found. Downloading and building rocm-cmake.")
    set(ROCM_CMAKE_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/rocm-cmake CACHE PATH "")
    download_project(
        PROJ           rocm-cmake
        GIT_REPOSITORY https://github.com/RadeonOpenCompute/rocm-cmake.git
        GIT_TAG        master
        INSTALL_DIR    ${ROCM_CMAKE_ROOT}
        CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        LOG_DOWNLOAD   TRUE
        LOG_CONFIGURE  TRUE
        LOG_BUILD      TRUE
        LOG_INSTALL    TRUE
        BUILD_PROJECT  TRUE
        ${UPDATE_DISCONNECTED_IF_AVAILABLE}
      )
    find_package(ROCM 0.6 REQUIRED CONFIG PATHS ${ROCM_CMAKE_ROOT})
  endif()
endif()

# rocm-cmake contains common cmake code for rocm projects to help
# setup and install
include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )
include( ROCMCheckTargetIds OPTIONAL )

