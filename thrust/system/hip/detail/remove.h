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
#include <thrust/system/hip/detail/copy_if.h>

BEGIN_NS_THRUST
namespace hip_rocprim
{

// in-place

template <class Derived, class InputIt, class StencilIt, class Predicate>
InputIt THRUST_HIP_FUNCTION
remove_if(execution_policy<Derived>& policy,
          InputIt                    first,
          InputIt                    last,
          StencilIt                  stencil,
          Predicate                  predicate)
{
    return hip_rocprim::copy_if(policy, first, last, stencil, first, detail::not1(predicate));
}

template <class Derived, class InputIt, class Predicate>
InputIt THRUST_HIP_FUNCTION
remove_if(execution_policy<Derived>& policy,
          InputIt                    first,
          InputIt                    last,
          Predicate                  predicate)
{
    return hip_rocprim::copy_if(policy, first, last, first, detail::not1(predicate));
}

template <class Derived, class InputIt, class T>
InputIt THRUST_HIP_FUNCTION
remove(execution_policy<Derived>& policy,
       InputIt                    first,
       InputIt                    last,
       const T&                   value)
{
    detail::equal_to_value<T> pred(value);
    return hip_rocprim::remove_if(policy, first, last, pred);
}

// copy

template <class Derived, class InputIt, class StencilIt, class OutputIt, class Predicate>
OutputIt THRUST_HIP_FUNCTION
remove_copy_if(execution_policy<Derived>& policy,
               InputIt                    first,
               InputIt                    last,
               StencilIt                  stencil,
               OutputIt                   result,
               Predicate                  predicate)
{
    return hip_rocprim::copy_if(policy, first, last, stencil, result, detail::not1(predicate));
}

template <class Derived, class InputIt, class OutputIt, class Predicate>
OutputIt THRUST_HIP_FUNCTION
remove_copy_if(execution_policy<Derived>& policy,
               InputIt                    first,
               InputIt                    last,
               OutputIt                   result,
               Predicate                  predicate)
{
    return hip_rocprim::copy_if(policy, first, last, result, detail::not1(predicate));
}

template <class Derived, class InputIt, class OutputIt, class T>
OutputIt THRUST_HIP_FUNCTION
remove_copy(execution_policy<Derived>& policy,
            InputIt                    first,
            InputIt                    last,
            OutputIt                   result,
            const T&                   value)
{
    detail::equal_to_value<T> pred(value);
    return hip_rocprim::remove_copy_if(policy, first, last, result, pred);
}

} // namespace hip_rocprim
END_NS_THRUST
#endif
