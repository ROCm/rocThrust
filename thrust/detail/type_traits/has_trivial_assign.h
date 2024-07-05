/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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


/*! \file type_traits.h
 *  \brief Temporarily define some type traits
 *         until nvcc can compile tr1::type_traits.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <cuda/std/type_traits>
#endif

THRUST_NAMESPACE_BEGIN

namespace detail
{

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
template<typename T>
struct has_trivial_assign
  : public integral_constant<
      bool,
      (is_pod<T>::value && !is_const<T>::value)
      || ::cuda::std::is_trivially_copy_assignable<T>::value
    >
{};
#else // THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA
template <typename T>
struct has_trivial_assign : public integral_constant<bool,
                                                     (is_pod<T>::value && !is_const<T>::value)
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
      || __is_trivially_assignable(T, const T)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
// only use the intrinsic for >= 4.3
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3)
      || __is_trivially_assignable(T, const T)
#endif // GCC VERSION
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
      || __is_trivially_assignable(T, const T)
#endif // THRUST_HOST_COMPILER
                                                     >
{};
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

} // end detail

THRUST_NAMESPACE_END
