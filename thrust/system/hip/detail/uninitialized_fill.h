/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <iterator>
#include <thrust/distance.h>
#include <thrust/system/hip/detail/execution_policy.h>
#include <thrust/system/hip/detail/parallel_for.h>
#include <thrust/system/hip/detail/util.h>

BEGIN_NS_THRUST
namespace hip_rocprim
{
namespace __uninitialized_fill
{
    template <class Iterator, class T>
    struct functor
    {
        Iterator items;
        T        value;

        typedef typename iterator_traits<Iterator>::value_type value_type;

        THRUST_HIP_FUNCTION
        functor(Iterator items_, T const& value_)
            : items(items_)
            , value(value_)
        {
        }

        template <class Size>
        void THRUST_HIP_DEVICE_FUNCTION operator()(Size idx)
        {
            value_type& out = raw_reference_cast(items[idx]);

            ::new(static_cast<void*>(&out)) value_type(value);
        }
    }; // struct functor
} // namespace __uninitialized_copy

template <class Derived, class Iterator, class Size, class T>
Iterator THRUST_HIP_FUNCTION
uninitialized_fill_n(execution_policy<Derived>& policy,
                     Iterator                   first,
                     Size                       count,
                     T const&                   x)
{
    typedef __uninitialized_fill::functor<Iterator, T> functor_t;

    hip_rocprim::parallel_for(policy, functor_t(first, x), count);
    return first + count;
}

template <class Derived, class Iterator, class T>
void THRUST_HIP_FUNCTION
uninitialized_fill(execution_policy<Derived>& policy,
                   Iterator                   first,
                   Iterator                   last,
                   T const&                   x)
{
    hip_rocprim::uninitialized_fill_n(policy, first, thrust::distance(first, last), x);
}

} // namespace hip_rocprim

END_NS_THRUST
#endif
