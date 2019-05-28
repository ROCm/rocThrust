/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditionu and the following disclaimer.
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

// XXX
// this file must not be included on its own, ever,
// but must be part of include in thrust/system/hip/detail/copy.h

#include <thrust/system/hip/config.h>

#include <thrust/advance.h>
#include <thrust/detail/dispatch/is_trivial_copy.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/temporary_buffer.h>
#include <thrust/distance.h>
#include <thrust/system/hip/detail/uninitialized_copy.h>

BEGIN_NS_THRUST
namespace hip_rocprim
{
namespace __copy
{
    template <class H, class D, class T, class Size>
    THRUST_HIP_HOST_FUNCTION void
    trivial_device_copy(thrust::cpp::execution_policy<H>&,
                        thrust::hip_rocprim::execution_policy<D>& device_s,
                        T*                                        dst,
                        T const*                                  src,
                        Size                                      count)
    {
        hipError_t status;
        status = hip_rocprim::trivial_copy_to_device(
            dst, src, count, hip_rocprim::stream(device_s));
        hip_rocprim::throw_on_error(status, "__copy::trivial_device_copy H->D: failed");
    }

    template <class D, class H, class T, class Size>
    THRUST_HIP_HOST_FUNCTION void
    trivial_device_copy(thrust::hip_rocprim::execution_policy<D>& device_s,
                        thrust::cpp::execution_policy<H>&         ,
                        T*                                        dst,
                        T const*                                  src,
                        Size                                      count)
    {
        hipError_t status;
        status = hip_rocprim::trivial_copy_from_device(
            dst, src, count, hip_rocprim::stream(device_s));
        hip_rocprim::throw_on_error(status, "trivial_device_copy D->H failed");
    }

    template <class System1, class System2, class InputIt, class Size, class OutputIt>
    OutputIt __host__ /* STREAMHPC WORKAROUND */ __device__
    cross_system_copy_n(thrust::execution_policy<System1>& sys1,
                        thrust::execution_policy<System2>& sys2,
                        InputIt                            begin,
                        Size                               n,
                        OutputIt                           result,
                        thrust::detail::true_type) // trivial copy
    {
        // STREAMHPC WORKAROUND
#if defined(THRUST_HIP_DEVICE_CODE)
        THRUST_UNUSED_VAR(sys1);
        THRUST_UNUSED_VAR(sys2);
        THRUST_UNUSED_VAR(n);
        THRUST_UNUSED_VAR(begin);
        return result;
#else
        typedef typename iterator_traits<InputIt>::value_type InputTy;
        trivial_device_copy(derived_cast(sys1),
                            derived_cast(sys2),
                            reinterpret_cast<InputTy*>(thrust::raw_pointer_cast(&*result)),
                            reinterpret_cast<InputTy const*>(thrust::raw_pointer_cast(&*begin)),
                            n);
        return result + n;
#endif
    }

    // non-trivial H->D copy
    template <class H, class D, class InputIt, class Size, class OutputIt>
    OutputIt __host__ /* STREAMHPC WORKAROUND */ __device__
    cross_system_copy_n(thrust::cpp::execution_policy<H>&         host_s,
                        thrust::hip_rocprim::execution_policy<D>& device_s,
                        InputIt                                   first,
                        Size                                      num_items,
                        OutputIt                                  result,
                        thrust::detail::false_type) // non-trivial copy
    {
        // get type of the input data
        typedef typename thrust::iterator_value<InputIt>::type InputTy;

        // STREAMHPC WORKAROUND
#if defined(THRUST_HIP_DEVICE_CODE)
        THRUST_UNUSED_VAR(host_s);
        THRUST_UNUSED_VAR(device_s);
        THRUST_UNUSED_VAR(first);
        THRUST_UNUSED_VAR(num_items);
        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
            (hip_rocprim::copy_n<D, InputTy*, Size, OutputIt>));
        return result;
#else
        // copy input data into host temp storage
        InputIt last = first;
        thrust::advance(last, num_items);
        //    thrust::detail::temporary_array<InputTy,H> temp(host_s, first, last);
        InputTy* temp = thrust::raw_pointer_cast(
            thrust::get_temporary_buffer<InputTy>(host_s, sizeof(InputTy) * num_items).first);

        for(Size idx = 0; idx != num_items; idx++)
        {
            ::new(static_cast<void*>(temp + idx)) InputTy(*first);
            ++first;
        }

        // allocate device temporary storage
        hipError_t status;
        InputTy*   d_in_ptr = thrust::raw_pointer_cast(
            thrust::get_temporary_buffer<InputTy>(device_s, sizeof(InputTy) * num_items).first);

        // trivial copy data from host to device
        status = hip_rocprim::trivial_copy_to_device(
            d_in_ptr, temp, num_items, hip_rocprim::stream(device_s));
        hip_rocprim::throw_on_error(status, "__copy:: H->D: failed");

        // device->device copy
        OutputIt ret = hip_rocprim::copy_n(device_s, d_in_ptr, num_items, result);

        // free device temporary storage
        thrust::return_temporary_buffer(host_s, temp);
        thrust::return_temporary_buffer(device_s, d_in_ptr);

        return ret;
#endif
    }

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
    // non-trivial copy D->H, only supported with HCC compiler
    // because copy ctor must have  __device__ annotations, which is hcc-only
    // feature
    template <class D, class H, class InputIt, class Size, class OutputIt>
    OutputIt __host__ /* STREAMHPC WORKAROUND */ __device__
    cross_system_copy_n(thrust::hip_rocprim::execution_policy<D>& device_s,
                        thrust::cpp::execution_policy<H>&         host_s,
                        InputIt                                   first,
                        Size                                      num_items,
                        OutputIt                                  result,
                        thrust::detail::false_type) // non-trivial copy

    {
        // get type of the input data
        typedef typename thrust::iterator_value<InputIt>::type InputTy;

        // STREAMHPC WORKAROUND
#if defined(THRUST_HIP_DEVICE_CODE)
        THRUST_UNUSED_VAR(device_s);
        THRUST_UNUSED_VAR(host_s);
        THRUST_UNUSED_VAR(first);
        THRUST_UNUSED_VAR(num_items);

        THRUST_HIP_PRESERVE_KERNELS_WORKAROUND(
            (hip_rocprim::uninitialized_copy_n<D, InputIt, Size, InputTy*>)
        );
        return result;
#else
        // allocate device temp storage
        hipError_t status;

        InputTy* d_in_ptr = thrust::raw_pointer_cast(
            thrust::get_temporary_buffer<InputTy>(device_s, sizeof(InputTy) * num_items).first);

        // uninitialize copy into temp device storage
        hip_rocprim::uninitialized_copy_n(device_s, first, num_items, d_in_ptr);

        // allocate host temp storage
        //    thrust::detail::temporary_array<InputTy,H> temp(0, host_s, num_items);
        InputTy* temp = thrust::raw_pointer_cast(
            thrust::get_temporary_buffer<InputTy>(host_s, num_items).first);

        // trivial copy from device to host
        status = hip_rocprim::trivial_copy_from_device(
            temp, d_in_ptr, num_items, hip_rocprim::stream(device_s));
        hip_rocprim::throw_on_error(status, "__copy:: D->H: failed");

        // copy host->host
        OutputIt ret = result;
        for(Size idx = 0; idx != num_items; ++idx)
        {
            // XXX generates warning using VC14 is there is type narrowing
            *ret = temp[idx];
            ++ret;
        }
        //OutputIt ret = thrust::copy(host_s, temp, temp+num_items, result);

        // free temp device storage
        thrust::return_temporary_buffer(device_s, d_in_ptr);
        thrust::return_temporary_buffer(host_s, temp);

        return ret;
#endif
    }
#endif

    template <class System1, class System2, class InputIt, class Size, class OutputIt>
    OutputIt __host__ /* STREAMHPC WORKAROUND */ __device__
    cross_system_copy_n(cross_system<System1, System2> systems, InputIt begin, Size n, OutputIt result)
    {
        return cross_system_copy_n(
            derived_cast(systems.sys1),
            derived_cast(systems.sys2),
            begin,
            n,
            result,
            typename thrust::detail::dispatch::is_trivial_copy<InputIt, OutputIt>::type());
    }

    template <class System1, class System2, class InputIterator, class OutputIterator>
    OutputIterator __host__ /* STREAMHPC WORKAROUND */ __device__
    cross_system_copy(cross_system<System1, System2> systems,
                      InputIterator                  begin,
                      InputIterator                  end,
                      OutputIterator                 result)
    {
        return cross_system_copy_n(systems, begin, thrust::distance(begin, end), result);
    }
} // namespace __copy
} // namespace hip_rocprim
END_NS_THRUST
