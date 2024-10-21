/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2024, Advanced Micro Devices, Inc.  All rights reserved.
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

#include <thrust/detail/config.h>

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP

#include <thrust/detail/alignment.h>
#include <thrust/detail/cstdint.h>
#include <thrust/detail/minmax.h>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/system/hip/config.h>
#include <thrust/system/hip/detail/get_value.h>
#include <thrust/system/hip/detail/general/temp_storage.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/util.h>

// rocprim include
#include <rocprim/rocprim.hpp>

THRUST_NAMESPACE_BEGIN

// forward declare generic reduce
// to circumvent circular dependency
template <typename DerivedPolicy,
          typename InputIterator,
          typename T,
          typename BinaryFunction>
THRUST_HOST_DEVICE
T reduce(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
         InputIterator                                               first,
         InputIterator                                               last,
         T                                                           init,
         BinaryFunction                                              binary_op);

namespace hip_rocprim
{
namespace __reduce
{
    template <typename Derived,
              typename InputIt,
              typename Size,
              typename T,
              typename BinaryOp>
    THRUST_HIP_RUNTIME_FUNCTION
    T reduce(execution_policy<Derived>& policy, InputIt first, Size num_items, T init, BinaryOp binary_op)
    {
        using namespace thrust::system::hip_rocprim::temp_storage;
        if(num_items == 0)
            return init;


        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(rocprim::reduce(NULL,
                                                    temp_storage_bytes,
                                                    first,
                                                    reinterpret_cast<T*>(NULL),
                                                    init,
                                                    static_cast<size_t>(num_items),
                                                    binary_op,
                                                    stream,
                                                    debug_sync),
                                    "reduce failed on 1st step");

        size_t storage_size;
        void*  ptr       = nullptr;
        void*  temp_stor = nullptr;
        T*     d_ret_ptr;

        auto l_part = make_linear_partition(make_partition(&temp_stor, temp_storage_bytes),
                                            ptr_aligned_array(&d_ret_ptr, 1));

        // Calculate storage_size including alignment
        hip_rocprim::throw_on_error(partition(ptr, storage_size, l_part));

        // Allocate temporary storage.
        thrust::detail::temporary_array<thrust::detail::uint8_t, Derived> tmp(policy, storage_size);
        ptr = static_cast<void*>(tmp.data().get());

        // Create pointers with alignment
        hip_rocprim::throw_on_error(partition(ptr, storage_size, l_part));

        hip_rocprim::throw_on_error(rocprim::reduce(ptr,
                                                    temp_storage_bytes,
                                                    first,
                                                    d_ret_ptr,
                                                    init,
                                                    static_cast<size_t>(num_items),
                                                    binary_op,
                                                    stream,
                                                    debug_sync),
                                    "reduce failed on 2nd step");
        T return_value = hip_rocprim::get_value(policy, d_ret_ptr);

        return return_value;
    }
}

//-------------------------
// Thrust API entry points
//-------------------------

template <class Derived, class InputIt, class Size, class T, class BinaryOp>
THRUST_HIP_FUNCTION
T reduce_n(execution_policy<Derived>& policy,
           InputIt                    first,
           Size                       num_items,
           T                          init,
           BinaryOp                   binary_op)
{
    // struct workaround is required for HIP-clang
    struct workaround
    {
        THRUST_HOST static T par(execution_policy<Derived>& policy,
                              InputIt                    first,
                              Size                       num_items,
                              T                          init,
                              BinaryOp                   binary_op)
        {
            return __reduce::reduce(policy, first, num_items, init, binary_op);
        }
        THRUST_DEVICE static T seq(execution_policy<Derived>& policy,
                                InputIt                    first,
                                Size                       num_items,
                                T                          init,
                                BinaryOp                   binary_op)
        {
            return thrust::reduce(
                cvt_to_seq(derived_cast(policy)), first, first + num_items, init, binary_op);
        }
  };

  #if __THRUST_HAS_HIPRT__
    return workaround::par(policy, first, num_items, init, binary_op);
  #else
    return workaround::seq(policy, first, num_items, init, binary_op);
  #endif


}

template <class Derived, class InputIt, class T, class BinaryOp>
THRUST_HIP_FUNCTION
T reduce(execution_policy<Derived>& policy,
         InputIt                    first,
         InputIt                    last,
         T                          init,
         BinaryOp                   binary_op)
{
    typedef typename iterator_traits<InputIt>::difference_type size_type;
    // FIXME: Check for RA iterator.
    size_type num_items = static_cast<size_type>(thrust::distance(first, last));
    return hip_rocprim::reduce_n(policy, first, num_items, init, binary_op);
}

template <class Derived, class InputIt, class T>
THRUST_HIP_FUNCTION
T reduce(execution_policy<Derived>& policy,
         InputIt                    first,
         InputIt                    last,
         T                          init)
{
    return hip_rocprim::reduce(policy, first, last, init, plus<T>());
}

template <class Derived, class InputIt>
THRUST_HIP_FUNCTION
typename iterator_traits<InputIt>::value_type
reduce(execution_policy<Derived>& policy,
       InputIt                    first,
       InputIt                    last)
{
    typedef typename iterator_traits<InputIt>::value_type value_type;
    return hip_rocprim::reduce(policy, first, last, value_type(0));
}

} // namespace  hip_rocprim

THRUST_NAMESPACE_END

#include <thrust/memory.h>
#include <thrust/reduce.h>

#endif
