/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019, Advanced Micro Devices, Inc.  All rights reserved.
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
#include <thrust/system/hip/config.h>

#include <thrust/detail/function.h>
#include <thrust/distance.h>
#include <thrust/system/hip/detail/parallel_for.h>
#include <thrust/system/hip/detail/util.h>

BEGIN_NS_THRUST

namespace hip_rocprim
{

// for_each functor
template <class Input, class UnaryOp>
struct for_each_f
{
    Input   input;
    UnaryOp op;

    THRUST_HIP_FUNCTION
    for_each_f(Input input, UnaryOp op)
        : input(input)
        , op(op)
    {
    }

    template <class Size>
    THRUST_HIP_FUNCTION void operator()(Size idx)
    {
        op(raw_reference_cast(input[idx]));
    }
};

//-------------------------
// Thrust API entry points
//-------------------------

// for_each_n
template <class Derived, class Input, class Size, class UnaryOp>
Input THRUST_HIP_FUNCTION
for_each_n(execution_policy<Derived>& policy, Input first, Size count, UnaryOp op)
{
    typedef detail::wrapped_function<UnaryOp, void> wrapped_t;
    wrapped_t                                       wrapped_op(op);

    hip_rocprim::parallel_for(policy, for_each_f<Input, wrapped_t>(first, wrapped_op), count);
    return first + count;
}

// for_each
template <class Derived, class Input, class UnaryOp>
Input THRUST_HIP_FUNCTION
for_each(execution_policy<Derived>& policy, Input first, Input last, UnaryOp op)
{
    typedef typename iterator_traits<Input>::difference_type size_type;
    size_type count = static_cast<size_type>(thrust::distance(first, last));
    return hip_rocprim::for_each_n(policy, first, count, op);
}

} // namespace hip_rocprim

END_NS_THRUST
#endif
