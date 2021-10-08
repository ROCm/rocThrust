# ########################################################################
# Copyright 2021 Advanced Micro Devices, Inc.
# ########################################################################

# ###########################
# rocThrust dependencies
# ###########################

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

# Find or download/install rocm-cmake project
find_package(ROCM 0.6 QUIET CONFIG PATHS ${ROCM_PATH})
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
