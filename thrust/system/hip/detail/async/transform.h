/******************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright© 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

// TODO: Move into system::hip

#pragma once

#include <thrust/detail/config.h>

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP

#include <thrust/system/hip/config.h>

#include <thrust/system/hip/detail/async/customization.h>
#include <thrust/system/hip/detail/parallel_for.h>
#include <thrust/system/hip/future.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/distance.h>
#include <thrust/advance.h>

#include <type_traits>

THRUST_NAMESPACE_BEGIN

namespace system { namespace hip { namespace detail
{

template <typename ForwardIt, typename OutputIt, typename UnaryOperation>
struct async_transform_fn
{
  ForwardIt first_;
  OutputIt output_;
  UnaryOperation op_;

  __host__ __device__
  async_transform_fn(ForwardIt&& first, OutputIt&& output, UnaryOperation&& op)
    : first_(std::move(first)), output_(std::move(output)), op_(std::move(op))
  {}

  template <typename Index>
  __host__ __device__
  void operator()(Index idx)
  {
    output_[idx] = op_(thrust::raw_reference_cast(first_[idx]));
  }
};

template <
  typename DerivedPolicy
, typename ForwardIt, typename Size, typename OutputIt, typename UnaryOperation
>
auto async_transform_n(
  execution_policy<DerivedPolicy>& policy,
  ForwardIt                        first,
  Size                             n,
  OutputIt                         output,
  UnaryOperation                   op
) -> unique_eager_event
{
  unique_eager_event e;

  // Set up stream with dependencies.

  hipStream_t const user_raw_stream = thrust::hip_rocprim::stream(policy);

  if (thrust::hip_rocprim::default_stream() != user_raw_stream)
  {
    e = make_dependent_event(
      std::tuple_cat(
        std::make_tuple(
          unique_stream(nonowning, user_raw_stream)
        )
      , extract_dependencies(
          std::move(thrust::detail::derived_cast(policy))
        )
      )
    );
  }
  else
  {
    e = make_dependent_event(
      extract_dependencies(
        std::move(thrust::detail::derived_cast(policy))
      )
    );
  }

  if( n == 0)
  {
    e.ready();
    return e;
  }

  // Run transform.

  async_transform_fn<ForwardIt, OutputIt, UnaryOperation> wrapped(
    std::move(first), std::move(output), std::move(op)
  );

  thrust::hip_rocprim::throw_on_error(
    thrust::hip_rocprim::__parallel_for::parallel_for(
      n, std::move(wrapped), e.stream().native_handle()
    )
  , "after transform launch"
  );

  return e;
}

}}} // namespace system::hip::detail

namespace hip_rocprim
{

// ADL entry point.
template <
  typename DerivedPolicy
, typename ForwardIt, typename Sentinel, typename OutputIt
, typename UnaryOperation
>
auto async_transform(
  execution_policy<DerivedPolicy>& policy,
  ForwardIt                        first,
  Sentinel                         last,
  OutputIt                         output,
  UnaryOperation&&                 op
)
THRUST_RETURNS(
  thrust::system::hip::detail::async_transform_n(
    policy, first, thrust::distance(first, last), output, THRUST_FWD(op)
  )
);

} // hip_rocprim

THRUST_NAMESPACE_END

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
