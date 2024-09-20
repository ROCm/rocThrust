/*
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

/*! \file thrust/system/hip/hipstdpar/copy.hpp
 *  \brief <tt>Copy operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#include "hipstd.hpp"

#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <execution>
#include <utility>

namespace std
{
    // BEGIN COPY
    template<
        typename I,
        typename O,
        enable_if_t<::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    O copy(execution::parallel_unsequenced_policy, I fi, I li, O fo)
    {
        return ::thrust::copy(::thrust::device, fi, li, fo);
    }

    template<
        typename I,
        typename O,
        enable_if_t<!::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    O copy(execution::parallel_unsequenced_policy, I fi, I li, O fo)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category,
            typename iterator_traits<O>::iterator_category>();

        return ::std::copy(::std::execution::par, fi, li, fo);
    }
    // END COPY

    // BEGIN COPY_IF
    template<
        typename I,
        typename O,
        typename P,
        enable_if_t<
            ::hipstd::is_offloadable_iterator<I, O>() &&
            ::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    O copy_if(execution::parallel_unsequenced_policy, I fi, I li, O fo, P p)
    {
        return
            ::thrust::copy_if(::thrust::device, fi, li, fo, ::std::move(p));
    }

    template<
        typename I,
        typename O,
        typename P,
        enable_if_t<
            !::hipstd::is_offloadable_iterator<I, O>() ||
            !::hipstd::is_offloadable_callable<P>()>* = nullptr>
    inline
    O copy_if(execution::parallel_unsequenced_policy, I fi, I li, O fo, P p)
    {
        if constexpr (!::hipstd::is_offloadable_iterator<I, O>()) {
            ::hipstd::unsupported_iterator_category<
                typename iterator_traits<I>::iterator_category,
                typename iterator_traits<O>::iterator_category>();
        }
        if constexpr (!::hipstd::is_offloadable_callable<P>()) {
            ::hipstd::unsupported_callable_type<P>();
        }

        return ::std::copy_if(
            ::std::execution::par, fi, li, fo, ::std::move(p));
    }
    // END COPY_IF

    // BEGIN COPY_N
    template<
        typename I,
        typename N,
        typename O,
        enable_if_t<::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    O copy_n(execution::parallel_unsequenced_policy, I fi, N n, O fo)
    {
        return ::thrust::copy_n(::thrust::device, fi, n, fo);
    }

    template<
        typename I,
        typename N,
        typename O,
        enable_if_t<!::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    O copy_n(execution::parallel_unsequenced_policy, I fi, N n, O fo)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category,
            typename iterator_traits<O>::iterator_category>();

        return ::std::copy_n(::std::execution::par, fi, n, fo);
    }
    // END COPY_N

    // BEGIN MOVE
    template<
        typename I,
        typename O,
        enable_if_t<::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    O move(execution::parallel_unsequenced_policy, I fi, I li, O fo)
    {
        return ::thrust::copy(
            ::thrust::device,
            make_move_iterator(fi),
            make_move_iterator(li),
            fo);
    }

    template<
        typename I,
        typename O,
        enable_if_t<!::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
    inline
    O move(execution::parallel_unsequenced_policy, I fi, I li, O fo)
    {
        ::hipstd::unsupported_iterator_category<
            typename iterator_traits<I>::iterator_category,
            typename iterator_traits<O>::iterator_category>();

        return ::std::move(::std::execution::par, fi, li, fo);
    }
    // END MOVE
}
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
