/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *  Modifications Copyright© 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

 #include <thrust/detail/seq.h>
 #include <thrust/system/cpp/detail/par.h>
 #include <thrust/system/hip/detail/par.h>
 #include <thrust/system/omp/detail/par.h>
 #include <thrust/system/tbb/detail/par.h>

#include "test_header.hpp"

template<typename T>
struct test_allocator_t
{
};

test_allocator_t<int> test_allocator = test_allocator_t<int>();
const test_allocator_t<int> const_test_allocator = test_allocator_t<int>();

struct test_memory_resource_t final : thrust::mr::memory_resource<>
{
    void * do_allocate(std::size_t size, std::size_t) override
    {
        return reinterpret_cast<void *>(size);
    }

    void do_deallocate(void * ptr, std::size_t size, std::size_t) override
    {
        ASSERT_EQ(ptr, reinterpret_cast<void *>(size));
    }
} test_memory_resource;

template<typename Policy, template <typename> class CRTPBase>
struct policy_info
{
    typedef Policy policy;

    template<template <typename, template <typename> class> class Template, typename Argument>
    struct apply_base_second
    {
        typedef Template<Argument, CRTPBase> type;
    };
};

template<typename PolicyInfo>
struct TestAllocatorAttachment
{
    template<typename Expected, typename T>
    static void assert_correct(T)
    {
        ASSERT_EQ(
            (thrust::detail::is_same<
                T,
                typename PolicyInfo::template apply_base_second<
                    thrust::detail::execute_with_allocator,
                    Expected
                >::type
            >::value), true);
    }

    template<typename ExpectedResource, typename T>
    static void assert_npa_correct(T)
    {
        ASSERT_EQ(
            (thrust::detail::is_same<
                T,
                typename PolicyInfo::template apply_base_second<
                    thrust::detail::execute_with_allocator,
                    thrust::mr::allocator<
                        thrust::detail::max_align_t,
                        ExpectedResource
                    >
                >::type
            >::value), true);
    }

    template<typename Policy>
    void test_temporary_allocation_valid(Policy policy)
    {
        using thrust::detail::get_temporary_buffer;

        return_temporary_buffer(
            policy,
            get_temporary_buffer<int>(
                policy,
                123
            ).first,
            123
        );
    }
    void operator()()
    {
        typename PolicyInfo::policy policy;

        // test correctness of attachment
        assert_correct<test_allocator_t<int> >(policy(test_allocator_t<int>()));
        assert_correct<test_allocator_t<int>&>(policy(test_allocator));
        assert_correct<test_allocator_t<int> >(policy(const_test_allocator));

        assert_npa_correct<test_memory_resource_t>(policy(&test_memory_resource));
        // test whether the resulting policy is actually usable
        // a real allocator is necessary here, unlike above
        std::allocator<int> alloc;
        const std::allocator<int> const_alloc;

        test_temporary_allocation_valid(policy(std::allocator<int>()));
        test_temporary_allocation_valid(policy(alloc));
        test_temporary_allocation_valid(policy(const_alloc));
        test_temporary_allocation_valid(policy(&test_memory_resource));

        test_temporary_allocation_valid(policy(std::allocator<int>()).after(1));
        test_temporary_allocation_valid(policy(alloc).after(1));
        test_temporary_allocation_valid(policy(const_alloc).after(1));
    }
};

typedef policy_info<
    thrust::detail::seq_t,
    thrust::system::detail::sequential::execution_policy
> sequential_info;
typedef policy_info<
    thrust::system::cpp::detail::par_t,
    thrust::system::cpp::detail::execution_policy
> cpp_par_info;
typedef policy_info<
    thrust::system::hip::detail::par_t,
    thrust::hip_rocprim::execute_on_stream_base
> hip_par_info;
typedef policy_info<
    thrust::system::omp::detail::par_t,
    thrust::system::omp::detail::execution_policy
> omp_par_info;
typedef policy_info<
    thrust::system::tbb::detail::par_t,
    thrust::system::tbb::detail::execution_policy
> tbb_par_info;

typedef ::testing::Types<Params<sequential_info>,
                         Params<cpp_par_info>,
                         Params<hip_par_info>,
                         Params<omp_par_info>,
                         Params<tbb_par_info>>
    PolicyTestsParams;

TESTS_DEFINE(AllocatorAwarePoliciesTests, PolicyTestsParams);

TYPED_TEST(AllocatorAwarePoliciesTests, TestAllocatorAttachmentInstance)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestAllocatorAttachment<T> test;
  test();
}
