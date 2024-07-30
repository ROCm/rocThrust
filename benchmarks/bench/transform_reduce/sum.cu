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

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <cstdlib>
#include <string>
#include <vector>

struct sum
{
    template <typename T, typename Policy = thrust::detail::device_t>
    float64_t run(thrust::device_vector<T> input)
    {
        bench_utils::gpu_timer d_timer;

        d_timer.start(0);
        bench_utils::do_not_optimize(thrust::reduce(Policy {}, input.begin(), input.end()));
        d_timer.stop(0);

        return d_timer.get_duration();
    }
};

template <class Benchmark, class T>
void run_benchmark(benchmark::State& state, const std::size_t elements, const std::string seed_type)
{
    // Benchmark object
    Benchmark benchmark {};

    // GPU times
    std::vector<double> gpu_times;

    // Generate input
    thrust::device_vector<T> input = bench_utils::generate(elements, seed_type);

    for(auto _ : state)
    {
        float64_t duration = benchmark.template run<T>(input);
        state.SetIterationTime(duration);
        gpu_times.push_back(duration);
    }

    // BytesProcessed include read and written bytes, so when the BytesProcessed/s are reported
    // it will actually be the global memory bandwidth gotten.
    state.SetBytesProcessed(state.iterations() * (elements + 1) * sizeof(T));
    state.SetItemsProcessed(state.iterations() * elements);

    const double gpu_cv         = bench_utils::StatisticsCV(gpu_times);
    state.counters["gpu_noise"] = gpu_cv;
}

#define CREATE_BENCHMARK(T, Elements)                                                        \
    benchmark::RegisterBenchmark(                                                            \
        bench_utils::bench_naming::format_name("{algo:transform_reduce,subalgo:" + name      \
                                               + ",input_type:" #T + ",elements:" #Elements) \
            .c_str(),                                                                        \
        run_benchmark<Benchmark, T>,                                                         \
        Elements,                                                                            \
        seed_type)

#define BENCHMARK_TYPE(type)                                          \
    CREATE_BENCHMARK(type, 1 << 16), CREATE_BENCHMARK(type, 1 << 20), \
        CREATE_BENCHMARK(type, 1 << 24), CREATE_BENCHMARK(type, 1 << 28)

template <class Benchmark>
void add_benchmarks(const std::string&                            name,
                    std::vector<benchmark::internal::Benchmark*>& benchmarks,
                    const std::string                             seed_type)
{
    std::vector<benchmark::internal::Benchmark*> bs
        = { BENCHMARK_TYPE(int8_t),
            BENCHMARK_TYPE(int16_t),
            BENCHMARK_TYPE(int32_t),
            BENCHMARK_TYPE(int64_t)
#if THRUST_BENCHMARKS_HAVE_INT128_SUPPORT
                ,
            BENCHMARK_TYPE(int128_t)
#endif
                ,
            BENCHMARK_TYPE(float32_t),
            BENCHMARK_TYPE(float64_t) };

    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<int>("repetitions", "repetitions", 5, "number of repetitions");
    parser.set_optional<std::string>(
        "name_format", "name_format", "human", "either: json,human,txt");
    parser.set_optional<std::string>("seed", "seed", "random", bench_utils::get_seed_message());
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const int repetitions = parser.get<int>("repetitions");
    bench_utils::bench_naming::set_format(
        parser.get<std::string>("name_format")); /* either: json,human*/
    const std::string seed_type = parser.get<std::string>("seed");
    parser.set_optional<std::string>("seed", "seed", "random", bench_utils::get_seed_message());
    parser.run_and_exit_if_error();


    // Add benchmark
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    add_benchmarks<sum>("sum", benchmarks, seed_type);

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMicrosecond);
        b->MinTime(0.5); // in seconds
        b->Repetitions(repetitions);
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks(new bench_utils::CustomReporter);

    // Finish
    benchmark::Shutdown();
    return 0;
}
