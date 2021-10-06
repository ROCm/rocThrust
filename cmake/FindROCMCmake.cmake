# ########################################################################
# Copyright 2021 Advanced Micro Devices, Inc.
# ########################################################################

# ###########################
# rocThrust dependencies
# ###########################

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

if (WIN32)
  find_package(ROCM 0.6 CONFIG QUIET PATHS ${ROCM_PATH})
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
endif()
