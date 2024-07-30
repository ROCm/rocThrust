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
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <cstdlib>
#include <string>
#include <vector>

struct basic
{
    template <typename T, typename Policy = thrust::detail::device_t>
    float64_t
    run(thrust::device_vector<T> input, thrust::device_vector<T> output, const std::size_t elements)
    {
        bench_utils::gpu_timer d_timer;

        d_timer.start(0);
        thrust::lower_bound(Policy {},
                            input.begin(),
                            input.begin() + elements,
                            input.begin() + elements,
                            input.end(),
                            output.begin());
        d_timer.stop(0);

        return d_timer.get_duration();
    }
};

template <class Benchmark, class T>
void run_benchmark(benchmark::State& state,
                   const std::size_t elements,
                   const std::string seed_type,
                   const std::size_t needles_ratio)
{
    // Benchmark object
    Benchmark benchmark {};

    // GPU times
    std::vector<double> gpu_times;

    // Generate input
    const auto needles
        = needles_ratio * static_cast<std::size_t>(static_cast<double>(elements) / 100.0f);

    thrust::device_vector<T> input = bench_utils::generate(elements + needles, seed_type);
    thrust::device_vector<T> output(needles);
    thrust::sort(input.begin(), input.begin() + elements);

    for(auto _ : state)
    {
        float64_t duration = benchmark.template run<T>(input, output, elements);
        state.SetIterationTime(duration);
        gpu_times.push_back(duration);
    }

    state.SetBytesProcessed(0);
    state.SetItemsProcessed(state.iterations() * needles);

    const double gpu_cv         = bench_utils::StatisticsCV(gpu_times);
    state.counters["gpu_noise"] = gpu_cv;
}

#define CREATE_BENCHMARK(T, Elements, NeedlesRatio)                                                \
    benchmark::RegisterBenchmark(bench_utils::bench_naming::format_name(                           \
                                     "{algo:vectorized_search,subalgo:" + name + ",input_type:" #T \
                                     + ",elements:" #Elements + ",needles_ratio:" #NeedlesRatio)   \
                                     .c_str(),                                                     \
                                 run_benchmark<Benchmark, T>,                                      \
                                 Elements,                                                         \
                                 seed_type,                                                        \
                                 NeedlesRatio)

#define BENCHMARK_ELEMENTS(type, elements)                                     \
    CREATE_BENCHMARK(type, elements, 1), CREATE_BENCHMARK(type, elements, 25), \
        CREATE_BENCHMARK(type, elements, 50)

#define BENCHMARK_TYPE(type)                                              \
    BENCHMARK_ELEMENTS(type, 1 << 16), BENCHMARK_ELEMENTS(type, 1 << 20), \
        BENCHMARK_ELEMENTS(type, 1 << 24), BENCHMARK_ELEMENTS(type, 1 << 28)

template <class Benchmark>
void add_benchmarks(const std::string&                            name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string                             seed_type)
{
    std::vector<benchmark::internal::Benchmark*> bs = {BENCHMARK_TYPE(int8_t),
                                                       BENCHMARK_TYPE(int16_t),
                                                       BENCHMARK_TYPE(int32_t),
                                                       BENCHMARK_TYPE(int64_t)};
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
    add_benchmarks<basic>("basic", benchmarks, seed_type);

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
