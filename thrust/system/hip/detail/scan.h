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
#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>
#include <thrust/functional.h>

#include <thrust/detail/minmax.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/distance.h>
#include <thrust/system/hip/detail/execution_policy.h>
#include <thrust/system/hip/detail/memory_buffer.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/util.h>

// rocprim include
#include <rocprim/rocprim.hpp>
#include <thrust/detail/alignment.h>
#include <thrust/detail/cstdint.h>

BEGIN_NS_THRUST
template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename AssociativeOperator>
OutputIterator THRUST_HIP_FUNCTION
inclusive_scan(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
               InputIterator                                               first,
               InputIterator                                               last,
               OutputIterator                                              result,
               AssociativeOperator                                         binary_op);

template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename T,
          typename AssociativeOperator>
OutputIterator THRUST_HIP_FUNCTION
exclusive_scan(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
               InputIterator                                               first,
               InputIterator                                               last,
               OutputIterator                                              result,
               T                                                           init,
               AssociativeOperator                                         binary_op);

namespace hip_rocprim
{
namespace __scan
{
    template <class Policy, class InputIt, class OutputIt, class Size, class ScanOp>
    OutputIt THRUST_HIP_RUNTIME_FUNCTION
    inclusive_scan(Policy&  policy,
                   InputIt  input_it,
                   OutputIt output_it,
                   Size     num_items,
                   ScanOp   scan_op)
    {
        if(num_items == 0)
            return output_it;

        void*       d_temp_storage     = nullptr;
        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::inclusive_scan(d_temp_storage,
                                                            temp_storage_bytes,
                                                            input_it,
                                                            output_it,
                                                            num_items,
                                                            scan_op,
                                                            stream,
                                                            debug_sync),
                                    "scan failed on 1st step");

        // Allocate temporary storage.
        d_temp_storage = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes);
        hip_rocprim::throw_on_error(hipGetLastError(), "scan failed to get memory buffer");

        // Run scan.
        hip_rocprim::throw_on_error(rocprim::inclusive_scan(d_temp_storage,
                                                            temp_storage_bytes,
                                                            input_it,
                                                            output_it,
                                                            num_items,
                                                            scan_op,
                                                            stream,
                                                            debug_sync),
                                    "scan failed on 2nd step");

        hip_rocprim::return_memory_buffer(policy, d_temp_storage);
        hip_rocprim::throw_on_error(hipGetLastError(), "scan failed to return memory buffer");

        return output_it + num_items;
    }

    template <class Policy, class InputIt, class OutputIt, class Size, class T, class ScanOp>
    OutputIt THRUST_HIP_RUNTIME_FUNCTION
    exclusive_scan(Policy&  policy,
                   InputIt  input_it,
                   OutputIt output_it,
                   Size     num_items,
                   T        init,
                   ScanOp   scan_op)
    {
        if(num_items == 0)
            return output_it;

        void*       d_temp_storage     = nullptr;
        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::exclusive_scan(d_temp_storage,
                                                            temp_storage_bytes,
                                                            input_it,
                                                            output_it,
                                                            init,
                                                            num_items,
                                                            scan_op,
                                                            stream,
                                                            debug_sync),
                                    "scan failed on 1st step");

        // Allocate temporary storage.
        d_temp_storage = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes);
        hip_rocprim::throw_on_error(hipGetLastError(), "scan failed to get memory buffer");

        // Run scan.
        hip_rocprim::throw_on_error(rocprim::exclusive_scan(d_temp_storage,
                                                            temp_storage_bytes,
                                                            input_it,
                                                            output_it,
                                                            init,
                                                            num_items,
                                                            scan_op,
                                                            stream,
                                                            debug_sync),
                                    "scan failed on 2nd step");

        hip_rocprim::return_memory_buffer(policy, d_temp_storage);
        hip_rocprim::throw_on_error(hipGetLastError(), "scan failed to return memory buffer");

        return output_it + num_items;
    }

} // namespace __scan

//-------------------------
// Thrust API entry points
//-------------------------

template <class Derived, class InputIt, class Size, class OutputIt, class ScanOp>
OutputIt THRUST_HIP_FUNCTION
inclusive_scan_n(execution_policy<Derived>& policy,
                 InputIt                    input_it,
                 Size                       num_items,
                 OutputIt                   result,
                 ScanOp                     scan_op)
{
    OutputIt ret = result;
    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
        (rocprim::inclusive_scan<rocprim::default_config, InputIt, OutputIt, ScanOp>)
    );
#if __THRUST_HAS_HIPRT__
    ret = __scan::inclusive_scan(policy, input_it, result, num_items, scan_op);
#else // __THRUST_HAS_HIPRT__
    ret = thrust::inclusive_scan(
        cvt_to_seq(derived_cast(policy)), input_it, input_it + num_items, result, scan_op
    );
#endif // __THRUST_HAS_HIPRT__
    return ret;
}

template <class Derived, class InputIt, class OutputIt, class ScanOp>
OutputIt THRUST_HIP_FUNCTION
inclusive_scan(execution_policy<Derived>& policy,
               InputIt                    first,
               InputIt                    last,
               OutputIt                   result,
               ScanOp                     scan_op)
{
    int num_items = static_cast<int>(thrust::distance(first, last));
    return hip_rocprim::inclusive_scan_n(policy, first, num_items, result, scan_op);
}

template <class Derived, class InputIt, class OutputIt>
OutputIt THRUST_HIP_FUNCTION
inclusive_scan(execution_policy<Derived>& policy,
               InputIt                    first,
               OutputIt                   last,
               OutputIt                   result)
{

    typedef typename thrust::detail::eval_if<thrust::detail::is_output_iterator<OutputIt>::value,
                                             thrust::iterator_value<InputIt>,
                                             thrust::iterator_value<OutputIt>>::type result_type;
    return inclusive_scan(policy, first, last, result, plus<result_type>());
}

__thrust_exec_check_disable__ template <class Derived,
                                        class InputIt,
                                        class Size,
                                        class OutputIt,
                                        class T,
                                        class ScanOp>
OutputIt THRUST_HIP_FUNCTION exclusive_scan_n(execution_policy<Derived>& policy,
                                              InputIt                    first,
                                              Size                       num_items,
                                              OutputIt                   result,
                                              T                          init,
                                              ScanOp                     scan_op)
{
    OutputIt ret = result;
    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
        (rocprim::exclusive_scan<rocprim::default_config, InputIt, OutputIt, T, ScanOp>));
#if __THRUST_HAS_HIPRT__

        ret = __scan::exclusive_scan(policy, first, result, num_items, init, scan_op);
#else // __THRUST_HAS_HIPRT__
        ret = thrust::exclusive_scan(
            cvt_to_seq(derived_cast(policy)), first, first + num_items, result, init, scan_op);
#endif // __THRUST_HAS_HIPRT__
        return ret;
    }

    template <class Derived, class InputIt, class OutputIt, class T, class ScanOp>
    OutputIt THRUST_HIP_FUNCTION
    exclusive_scan(execution_policy<Derived>& policy,
                   InputIt                    first,
                   InputIt                    last,
                   OutputIt                   result,
                   T                          init,
                   ScanOp                     scan_op)
    {
        int num_items = static_cast<int>(thrust::distance(first, last));
        return hip_rocprim::exclusive_scan_n(policy, first, num_items, result, init, scan_op);
    }

    template <class Derived, class InputIt, class OutputIt, class T>
    OutputIt THRUST_HIP_FUNCTION
    exclusive_scan(execution_policy<Derived>& policy,
                   InputIt                    first,
                   OutputIt                   last,
                   OutputIt                   result,
                   T                          init)
    {
        return exclusive_scan(policy, first, last, result, init, plus<T>());
    }

    template <class Derived, class InputIt, class OutputIt>
    OutputIt THRUST_HIP_FUNCTION
    exclusive_scan(execution_policy<Derived>& policy,
                   InputIt                    first,
                   OutputIt                   last,
                   OutputIt                   result)
    {
        typedef typename thrust::detail::eval_if<thrust::detail::is_output_iterator<OutputIt>::value,
                                                 thrust::iterator_value<InputIt>,
                                                 thrust::iterator_value<OutputIt>>::type result_type;
        return exclusive_scan(policy, first, last, result, result_type(0));
    }

} // namespace  hip_rocprim

END_NS_THRUST

#include <thrust/scan.h>

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
