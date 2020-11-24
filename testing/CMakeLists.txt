# ########################################################################
# Copyright 2019 - 2020 Advanced Micro Devices, Inc.
# ########################################################################

# Update flags to reflect RDC options. See note in ThrustCudaConfig.cmake --
# these flag variables behave unintuitively:
if (THRUST_ENABLE_TESTS_WITH_RDC)
  set(CMAKE_CUDA_FLAGS "${THRUST_CUDA_FLAGS_BASE} ${THRUST_CUDA_FLAGS_RDC}")
else()
  set(CMAKE_CUDA_FLAGS "${THRUST_CUDA_FLAGS_BASE} ${THRUST_CUDA_FLAGS_NO_RDC}")
endif()

# Generate testing framework libraries:
add_subdirectory(unittest)

# List of tests that aren't implemented for all backends, but are implemented for CUDA.
set(partially_implemented_CUDA
  async_copy
  async_for_each
  async_reduce
  async_reduce_into
  async_sort
  async_transform
  event
  future

  # This test is incompatible with TBB and OMP, since it requires special per-device
  # handling to process exceptions in a device function, which is only implemented
  # for CUDA.
  unittest_static_assert
)
target_link_libraries(testing_common
    INTERFACE
        rocthrust
        roc::rocprim_hip
        ${GTEST_BOTH_LIBRARIES}
)
foreach(amdgpu_target ${AMDGPU_TARGETS})
    target_link_libraries(testing_common
        INTERFACE
          --cuda-gpu-arch=${amdgpu_target}
    )
endforeach()

if(NOT CMAKE_VERSION VERSION_LESS 3.13)
    set_source_files_properties(unittest/testframework.cu
        PROPERTIES
            LANGUAGE CXX
    )
    add_library(testframework OBJECT unittest/testframework.cu)
else() # Workaround
    set_source_files_properties(unittest/testframework.cu
        PROPERTIES
            LANGUAGE CXX
    )
    add_library(testframework STATIC unittest/testframework.cu)
endif()
target_link_libraries(testframework
    PUBLIC
        testing_common
)

# List of tests that aren't implemented for all backends, but are implemented for OMP.
set(partially_implemented_OMP
)

# List of all partially implemented tests.
set(partially_implemented
  ${partially_implemented_CUDA}
  ${partially_implemented_CPP}
  ${partially_implemented_TBB}
  ${partially_implemented_OMP}
)
list(REMOVE_DUPLICATES partially_implemented)

## thrust_add_test
#
# Add a test executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the test
#   target. Useful for post-processing target information per-backend.
# test_name: The name of the test minus "<config_prefix>.test." For example,
#   testing/vector.cu will be "vector", and testing/cuda/copy.cu will be
#   "cuda.copy".
# test_src: The source file that implements the test.
# thrust_target: The reference thrust target with configuration information.
#
function(thrust_add_test target_name_var test_name test_src thrust_target)
  thrust_get_target_property(config_host ${thrust_target} HOST)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

  # Wrap the .cu file in .cpp for non-CUDA backends
  if ("CUDA" STREQUAL "${config_device}")
    set(real_test_src "${test_src}")
  else()
    thrust_wrap_cu_in_cpp(real_test_src "${test_src}" ${thrust_target})
  endif()

  # The actual name of the test's target:
  set(test_target ${config_prefix}.test.${test_name})
  set(${target_name_var} ${test_target} PARENT_SCOPE)

  # Related target names:
  set(config_framework_target ${config_prefix}.test.framework)
  set(config_meta_target ${config_prefix}.tests)
  set(test_meta_target thrust.all.test.${test_name})

  add_executable(${test_target} "${real_test_src}")
  target_link_libraries(${test_target} ${config_framework_target})
  target_include_directories(${test_target} PRIVATE "${Thrust_SOURCE_DIR}/testing")
  thrust_clone_target_properties(${test_target} ${thrust_target})

  # Add to the active configuration's meta target
  add_dependencies(${config_meta_target} ${test_target})

  # Meta target that builds tests with this name for all configurations:
  if (NOT TARGET ${test_meta_target})
    add_custom_target(${test_meta_target})
  endif()
  add_dependencies(${test_meta_target} ${test_target})

  add_test(NAME ${test_target}
    COMMAND "${CMAKE_COMMAND}"
    "-DTHRUST_BINARY=$<TARGET_FILE:${test_target}>"
    "-DTHRUST_SOURCE=${Thrust_SOURCE_DIR}"
    -P "${Thrust_SOURCE_DIR}/cmake/ThrustRunTest.cmake"
  )

  # Run OMP/TBB tests in serial. Multiple OMP processes will massively
  # oversubscribe the machine with GCC's OMP, and we want to test these with
  # the full CPU available to each unit test.
  set(config_systems ${config_host} ${config_device})
  if (("OMP" IN_LIST config_systems) OR ("TBB" IN_LIST config_systems))
    set_tests_properties(${test_target} PROPERTIES RUN_SERIAL ON)
  endif()

  # Check for per-test script. Script will be included in the current scope
  # to allow custom property modifications.
  get_filename_component(test_cmake_script "${test_src}" NAME_WLE)
  set(test_cmake_script "${CMAKE_CURRENT_LIST_DIR}/${test_cmake_script}.cmake")
  # Use a glob so we can detect if this changes:
  file(GLOB test_cmake_script
    RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
    CONFIGURE_DEPENDS
    "${test_cmake_script}"
  )
  if (test_cmake_script) # Will be non-empty only if the script exists
    include("${test_cmake_script}")
  endif()
endfunction()

# ****************************************************************************
# Tests
# ****************************************************************************

add_thrust_test("adjacent_difference")
add_thrust_test("advance")
add_thrust_test("alignment")
add_thrust_test("allocator")
add_thrust_test("allocator_aware_policies")
add_thrust_test("async_copy")
add_thrust_test("async_for_each")
add_thrust_test("async_reduce")
add_thrust_test("async_reduce_into")
add_thrust_test("async_sort")
add_thrust_test("async_transform")
add_thrust_test("binary_search")
add_thrust_test("binary_search_descending")
add_thrust_test("binary_search_vector")
#add_thrust_test("binary_search_vector_descending")
add_thrust_test("caching_allocator")
add_thrust_test("complex")
add_thrust_test("complex_transform")
add_thrust_test("constant_iterator")
#add_thrust_test("copy")
add_thrust_test("copy_n")
add_thrust_test("count")
add_thrust_test("counting_iterator")
add_thrust_test("cstdint")
add_thrust_test("dependencies_aware_policies")
add_thrust_test("dereference")
add_thrust_test("device_delete")
add_thrust_test("device_ptr")
add_thrust_test("device_reference")
add_thrust_test("discard_iterator")
add_thrust_test("distance")
add_thrust_test("equal")
add_thrust_test("event")
add_thrust_test("fill")
add_thrust_test("find")
add_thrust_test("for_each")
add_thrust_test("functional")
add_thrust_test("functional_arithmetic")
add_thrust_test("functional_bitwise")
add_thrust_test("functional_logical")
add_thrust_test("functional_placeholders_arithmetic")
add_thrust_test("functional_placeholders_bitwise")
add_thrust_test("functional_placeholders_compound_assignment")
add_thrust_test("functional_placeholders_logical")
add_thrust_test("functional_placeholders_miscellaneous")
add_thrust_test("functional_placeholders_relational")
add_thrust_test("future")
add_thrust_test("gather")
add_thrust_test("generate")
add_thrust_test("generate_const_iterators")
add_thrust_test("inner_product")
add_thrust_test("is_contiguous_iterator")
add_thrust_test("is_operator_function_object")
add_thrust_test("is_partitioned")
add_thrust_test("is_sorted")
add_thrust_test("is_sorted_until")
add_thrust_test("logical")
add_thrust_test("max_element")
add_thrust_test("memory")
add_thrust_test("merge")
add_thrust_test("merge_by_key")
add_thrust_test("merge_key_value")
add_thrust_test("metaprogamming")
add_thrust_test("minmax_element")
add_thrust_test("min_and_max")
add_thrust_test("min_element")
add_thrust_test("mismatch")
add_thrust_test("mr_disjoint_pool")
add_thrust_test("mr_new")
add_thrust_test("mr_pool")
add_thrust_test("mr_pool_options")
add_thrust_test("out_of_memory_recovery")
add_thrust_test("pair")
add_thrust_test("pair_reduce")
add_thrust_test("pair_scan")
add_thrust_test("pair_scan_by_key")
add_thrust_test("pair_sort")
add_thrust_test("pair_sort_by_key")
add_thrust_test("pair_transform")
add_thrust_test("partition")
add_thrust_test("partition_point")
add_thrust_test("permutation_iterator")
add_thrust_test("preprocessor")
add_thrust_test("random")
add_thrust_test("reduce")
add_thrust_test("reduce_by_key")
add_thrust_test("reduce_large")
add_thrust_test("remove")
add_thrust_test("replace")
add_thrust_test("reverse")
add_thrust_test("reverse_iterator")
add_thrust_test("scan")
#add_thrust_test("scan_by_key")
add_thrust_test("scatter")
add_thrust_test("sequence")
add_thrust_test("set_difference")
add_thrust_test("set_difference_by_key")
add_thrust_test("set_difference_by_key_descending")
add_thrust_test("set_difference_descending")
add_thrust_test("set_difference_key_value")
add_thrust_test("set_intersection")
add_thrust_test("set_intersection_by_key")
add_thrust_test("set_intersection_by_key_descending")
add_thrust_test("set_intersection_descending")
add_thrust_test("set_intersection_key_value")
add_thrust_test("set_symmetric_difference")
add_thrust_test("set_symmetric_difference_by_key")
add_thrust_test("set_symmetric_difference_by_key_descending")
add_thrust_test("set_symmetric_difference_descending")
add_thrust_test("set_union")
add_thrust_test("set_union_by_key")
add_thrust_test("set_union_by_key_descending")
add_thrust_test("set_union_descending")
add_thrust_test("set_union_key_value")
add_thrust_test("shuffle")
add_thrust_test("sort")
add_thrust_test("sort_by_key")
add_thrust_test("sort_by_key_variable_bits")
add_thrust_test("sort_permutation_iterator")
add_thrust_test("sort_variable_bits")
add_thrust_test("stable_sort")
add_thrust_test("stable_sort_by_key")
add_thrust_test("stable_sort_by_key_large")
add_thrust_test("stable_sort_large")
add_thrust_test("swap_ranges")
add_thrust_test("tabulate")
add_thrust_test("transform")
add_thrust_test("transform_iterator")
add_thrust_test("transform_input_output_iterator")
add_thrust_test("transform_output_iterator")
add_thrust_test("transform_reduce")
add_thrust_test("transform_scan")
add_thrust_test("trivial_sequence")
add_thrust_test("tuple")
add_thrust_test("tuple_algorithms")
add_thrust_test("tuple_reduce")
add_thrust_test("tuple_scan")
add_thrust_test("tuple_sort")
add_thrust_test("tuple_transform")
add_thrust_test("type_traits")
add_thrust_test("uninitialized_copy")
add_thrust_test("uninitialized_fill")
add_thrust_test("unique")
add_thrust_test("unique_by_key")
add_thrust_test("unittest_static_assert")
add_thrust_test("unittest_tester")
add_thrust_test("vector")
add_thrust_test("vector_allocators")
add_thrust_test("vector_insert")
add_thrust_test("vector_manipulation")
add_thrust_test("zip_function")
add_thrust_test("zip_iterator")
add_thrust_test("zip_iterator_reduce")
add_thrust_test("zip_iterator_reduce_by_key")
add_thrust_test("zip_iterator_scan")
add_thrust_test("zip_iterator_sort")
add_thrust_test("zip_iterator_sort_by_key")
