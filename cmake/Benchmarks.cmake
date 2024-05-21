# ########################################################################
# Copyright 2024 Advanced Micro Devices, Inc.
# ########################################################################

# ###########################
# rocThrust benchmarks
# ###########################

# Common functionality for configuring rocThrust's benchmarks

# Registers a .cu as C++ rocThrust benchmark
function(add_thrust_benchmark BENCHMARK_NAME BENCHMARK_SOURCE NOT_INTERNAL)
    set(BENCHMARK_TARGET "benchmark_trust_${BENCHMARK_NAME}")
    set_source_files_properties(${BENCHMARK_SOURCE}
        PROPERTIES
            LANGUAGE CXX
    )
    add_executable(${BENCHMARK_TARGET} ${BENCHMARK_SOURCE})

    target_link_libraries(${BENCHMARK_TARGET}
        PRIVATE
            rocthrust
            roc::rocprim_hip
    )
    endif()
    foreach(gpu_target ${GPU_TARGETS})
        target_link_libraries(${BENCHMARK_TARGET}
            INTERFACE
                --cuda-gpu-arch=${gpu_target}
        )
    endforeach()

    # Separate normal from internal benchmarks
    if(NOT_INTERNAL)
        set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/benchmarks/")
    else()
        set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/benchmarks/internal/")
    endif()

    set_target_properties(${BENCHMARK_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    )
    rocm_install(TARGETS ${BENCHMARK_TARGET} COMPONENT benchmarks)
endfunction()
