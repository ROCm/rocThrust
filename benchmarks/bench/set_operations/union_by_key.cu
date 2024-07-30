// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "./by_key.hpp"

// rocThrust
#include <thrust/set_operations.h>

// Google Benchmark
#include <benchmark/benchmark.h>

struct op_t
{
    static constexpr bool read_all_values = true;

    template <class DerivedPolicy,
              class InputIterator1,
              class InputIterator2,
              class InputIterator3,
              class InputIterator4,
              class OutputIterator1,
              class OutputIterator2>
    __host__ thrust::pair<OutputIterator1, OutputIterator2>
             operator()(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
               InputIterator1                                              keys_first1,
               InputIterator1                                              keys_last1,
               InputIterator2                                              keys_first2,
               InputIterator2                                              keys_last2,
               InputIterator3                                              values_first1,
               InputIterator4                                              values_first2,
               OutputIterator1                                             keys_result,
               OutputIterator2                                             values_result) const
    {
        return thrust::set_union_by_key(exec,
                                        keys_first1,
                                        keys_last1,
                                        keys_first2,
                                        keys_last2,
                                        values_first1,
                                        values_first2,
                                        keys_result,
                                        values_result);
    }
};

int main(int argc, char* argv[])
{
    benchmark::Initialize(&argc, argv);
    bench_utils::bench_naming::set_format("human"); /* either: json,human,txt*/

    // Benchmark parameters
    const std::string seed_type = "random";

    // Benchmark info
    bench_utils::add_common_benchmark_info();
    benchmark::AddCustomContext("seed", seed_type);

    // Add benchmark
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks<op_t>("union", benchmarks, seed_type);

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMicrosecond);
        b->MinTime(0.5); // in seconds
        b->Repetitions(5);
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks(new bench_utils::CustomReporter);

    // Finish
    benchmark::Shutdown();
    return 0;
}
