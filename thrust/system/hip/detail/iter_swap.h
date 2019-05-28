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

#pragma once

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <thrust/detail/config.h>
#include <thrust/system/hip/config.h>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/swap.h>
#include <thrust/system/hip/detail/execution_policy.h>

BEGIN_NS_THRUST
namespace hip_rocprim
{

template <typename DerivedPolicy, typename Pointer1, typename Pointer2>
void THRUST_HIP_FUNCTION
iter_swap(thrust::hip::execution_policy<DerivedPolicy>&, Pointer1 a, Pointer2 b)
{
#if defined(THRUST_HIP_DEVICE_CODE)
    Pointer2 (*fptr)(Pointer1, Pointer1, Pointer2) = thrust::swap_ranges;
    (void)fptr;

    using thrust::swap;
    swap(*thrust::raw_pointer_cast(a), *thrust::raw_pointer_cast(b));
#else
    thrust::swap_ranges(a, a + 1, b);
#endif
} // end iter_swap()

} // end hip_rocprim
END_NS_THRUST
#endif
