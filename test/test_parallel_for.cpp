/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/memory.h>

#include "test_header.hpp"

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
#define PARALLEL_FOR thrust::hip_rocprim::parallel_for
#elif THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
#define PARALLEL_FOR thrust::cuda_cub::parallel_for
#endif

#ifdef PARALLEL_FOR

TESTS_DEFINE(ParallelForTests, ::testing::Types<Params<char> >)

template <typename T, size_t limit>
struct mark_processed_functor
{
    T*       ptr;
    __host__ __device__ void operator()(size_t x)
    {
        if(x < limit)
        {
            ptr[x] += 1;
        }
    }
};

TYPED_TEST(ParallelForTests, HostPathSimpleTest)
{
    thrust::device_system_tag tag;
    using T = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    const size_t mem_limit = (1ull << 30) * 3;

    const std::vector<size_t> sizes = { 1ull << 31, (1ull << 30) * 3 - 100, (1ull << 32) * 3 };

    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        auto mem_size = std::min(size, mem_limit);
        auto ptr      = thrust::malloc<T>(tag, sizeof(T) * mem_size);
        auto raw_ptr  = thrust::raw_pointer_cast(ptr);
        if(size > 0)
            ASSERT_NE(raw_ptr, nullptr);

        // Zero input memory
        if(size > 0)
            HIP_CHECK(hipMemset(raw_ptr, 0, sizeof(T) * mem_size));

        // Create unary function
        mark_processed_functor<T, mem_limit> func;
        func.ptr = raw_ptr;

        // Run for_each in [0; end] range
        auto end      = size * 3 / 4;
        PARALLEL_FOR(tag, func, end);

        std::vector<T> output(size);
        HIP_CHECK(hipMemcpy(output.data(), raw_ptr, mem_size * sizeof(T), hipMemcpyDeviceToHost));

        for(size_t i = 0; i < mem_size; i++)
        {
            if(i < end)
            {
                ASSERT_EQ(output[i], T(1)) << "where index = " << i;
            }
            else
            {
                ASSERT_EQ(output[i], T(0)) << "where index = " << i;
            }
        }

        // Free
        thrust::free(tag, ptr);
    }
}

#undef PARALLEL_FOR
#endif
