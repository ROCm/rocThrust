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

// Benchmark utils
#include "../../bench_utils/bench_utils.hpp"

// rocThrust
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <cstdlib>
#include <string>
#include <vector>

struct by_key
{
    template <typename KeyT, typename ValueT, typename Policy = thrust::detail::device_t>
    float64_t run(thrust::device_vector<KeyT>   input_keys,
                  thrust::device_vector<ValueT> input_vals,
                  thrust::device_vector<KeyT>   output_keys,
                  thrust::device_vector<ValueT> output_vals)
    {
        bench_utils::gpu_timer d_timer;

        d_timer.start(0);
        thrust::unique_by_key_copy(Policy {},
                                   input_keys.cbegin(),
                                   input_keys.cend(),
                                   input_vals.cbegin(),
                                   output_keys.begin(),
                                   output_vals.begin());
        d_timer.stop(0);

        return d_timer.get_duration();
    }
};

template <class Benchmark, class KeyT, class ValueT>
void run_benchmark(benchmark::State& state,
                   const std::size_t elements,
                   const std::string seed_type,
                   const std::size_t max_segment_size)
{
    // Benchmark object
    Benchmark benchmark {};

    // GPU times
    std::vector<double> gpu_times;

    // Generate input
    constexpr std::size_t       min_segment_size = 1;
    thrust::device_vector<KeyT> input_keys       = bench_utils::generate.uniform.key_segments(
        elements, seed_type, min_segment_size, max_segment_size);
    thrust::device_vector<ValueT> input_vals(elements);

    // Output
    thrust::device_vector<KeyT> output_keys(elements);
    const std::size_t           unique_elements = thrust::distance(
        output_keys.begin(),
        thrust::unique_copy(input_keys.cbegin(), input_keys.cend(), output_keys.begin()));
    thrust::device_vector<ValueT> output_vals(unique_elements);

    for(auto _ : state)
    {
        float64_t duration = benchmark.template run<KeyT, ValueT>(
            input_keys, input_vals, output_keys, output_vals);
        state.SetIterationTime(duration);
        gpu_times.push_back(duration);
    }

    // BytesProcessed include read and written bytes, so when the BytesProcessed/s are reported
    // it will actually be the global memory bandwidth gotten.
    state.SetBytesProcessed(state.iterations() * (elements + unique_elements)
                            * (sizeof(KeyT) + sizeof(ValueT)));
    state.SetItemsProcessed(state.iterations() * elements);

    const double gpu_cv         = bench_utils::StatisticsCV(gpu_times);
    state.counters["gpu_noise"] = gpu_cv;
}

#define CREATE_BENCHMARK(KeyT, ValueT, Elements, MaxSegmentSize)                                   \
    benchmark::RegisterBenchmark(                                                                  \
        bench_utils::bench_naming::format_name("{algo:unique,subalgo:" + name + ",key_type:" #KeyT \
                                               + ",value_type:" #ValueT + ",elements:" #Elements   \
                                               + ",max_segment_size:" #MaxSegmentSize)             \
            .c_str(),                                                                              \
        run_benchmark<Benchmark, KeyT, ValueT>,                                                    \
        Elements,                                                                                  \
        seed_type,                                                                                 \
        MaxSegmentSize)

#define BENCHMARK_ELEMENTS(key_type, value_type, elements) \
    CREATE_BENCHMARK(key_type, value_type, elements, 1),   \
        CREATE_BENCHMARK(key_type, value_type, elements, 8)

#define BENCHMARK_VALUE_TYPE(key_type, value_type)         \
    BENCHMARK_ELEMENTS(key_type, value_type, 1 << 16),     \
        BENCHMARK_ELEMENTS(key_type, value_type, 1 << 20), \
        BENCHMARK_ELEMENTS(key_type, value_type, 1 << 24), \
        BENCHMARK_ELEMENTS(key_type, value_type, 1 << 28)

#if THRUST_BENCHMARKS_HAVE_INT128_SUPPORT
#define BENCHMARK_KEY_TYPE(key_type)                                                       \
    BENCHMARK_VALUE_TYPE(key_type, int8_t), BENCHMARK_VALUE_TYPE(key_type, int16_t),       \
        BENCHMARK_VALUE_TYPE(key_type, int32_t), BENCHMARK_VALUE_TYPE(key_type, int64_t),  \
        BENCHMARK_VALUE_TYPE(key_type, int64_t), BENCHMARK_VALUE_TYPE(key_type, int128_t), \
        BENCHMARK_VALUE_TYPE(key_type, float), BENCHMARK_VALUE_TYPE(key_type, double)
#else
#define BENCHMARK_KEY_TYPE(key_type)                                                      \
    BENCHMARK_VALUE_TYPE(key_type, int8_t), BENCHMARK_VALUE_TYPE(key_type, int16_t),      \
        BENCHMARK_VALUE_TYPE(key_type, int32_t), BENCHMARK_VALUE_TYPE(key_type, int64_t), \
        BENCHMARK_VALUE_TYPE(key_type, int64_t), BENCHMARK_VALUE_TYPE(key_type, float),   \
        BENCHMARK_VALUE_TYPE(key_type, double)
#endif

template <class Benchmark>
void add_benchmarks(const std::string&                            name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string                             seed_type)
{
    std::vector<benchmark::internal::Benchmark*> bs
        = { BENCHMARK_KEY_TYPE(int8_t),
            BENCHMARK_KEY_TYPE(int16_t),
            BENCHMARK_KEY_TYPE(int32_t),
            BENCHMARK_KEY_TYPE(int64_t)
#if THRUST_BENCHMARKS_HAVE_INT128_SUPPORT
                ,
            BENCHMARK_KEY_TYPE(int128_t)
#endif
          };
    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

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
    add_benchmarks<by_key>("by_key", benchmarks, seed_type);

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
