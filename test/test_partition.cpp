/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/count.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "test_header.hpp"

TESTS_DEFINE(PartitionTests, FullTestsParams);
TESTS_DEFINE(PartitionVectorTests, VectorSignedIntegerTestsParams);
TESTS_DEFINE(PartitionIntegerTests, IntegerTestsParams);

template <typename T>
struct is_even
{
    __host__ __device__ bool operator()(T x) const
    {
        return ((int)x % 2) == 0;
    }
};

TYPED_TEST(PartitionVectorTests, TestPartitionSimple)
{
    using Vector   = typename TestFixture::input_type;
    using T        = typename Vector::value_type;
    using Iterator = typename Vector::iterator;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector data(5);
    data[0] = 1;
    data[1] = 2;
    data[2] = 1;
    data[3] = 1;
    data[4] = 2;

    Iterator iter = thrust::partition(data.begin(), data.end(), is_even<T>());

    Vector ref(5);
    ref[0] = 2;
    ref[1] = 2;
    ref[2] = 1;
    ref[3] = 1;
    ref[4] = 1;

    ASSERT_EQ(iter - data.begin(), 2);
    ASSERT_EQ(data, ref);
}

TYPED_TEST(PartitionVectorTests, TestPartitionStencilSimple)
{
    using Vector   = typename TestFixture::input_type;
    using T        = typename Vector::value_type;
    using Iterator = typename Vector::iterator;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector data(5);
    data[0] = 0;
    data[1] = 1;
    data[2] = 0;
    data[3] = 0;
    data[4] = 1;

    Vector stencil(5);
    stencil[0] = 1;
    stencil[1] = 2;
    stencil[2] = 1;
    stencil[3] = 1;
    stencil[4] = 2;

    Iterator iter = thrust::partition(data.begin(), data.end(), stencil.begin(), is_even<T>());

    Vector ref(5);
    ref[0] = 1;
    ref[1] = 1;
    ref[2] = 0;
    ref[3] = 0;
    ref[4] = 0;

    ASSERT_EQ(iter - data.begin(), 2);
    ASSERT_EQ(data, ref);
}

TYPED_TEST(PartitionVectorTests, TestPartitionCopySimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector data(5);
    data[0] = 1;
    data[1] = 2;
    data[2] = 1;
    data[3] = 1;
    data[4] = 2;

    Vector true_results(2);
    Vector false_results(3);

    thrust::pair<typename Vector::iterator, typename Vector::iterator> ends
        = thrust::partition_copy(
            data.begin(), data.end(), true_results.begin(), false_results.begin(), is_even<T>());

    Vector true_ref(2);
    true_ref[0] = 2;
    true_ref[1] = 2;

    Vector false_ref(3);
    false_ref[0] = 1;
    false_ref[1] = 1;
    false_ref[2] = 1;

    ASSERT_EQ(2, ends.first - true_results.begin());
    ASSERT_EQ(3, ends.second - false_results.begin());
    ASSERT_EQ(true_ref, true_results);
    ASSERT_EQ(false_ref, false_results);
}

TYPED_TEST(PartitionVectorTests, TestPartitionCopyStencilSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector data(5);
    data[0] = 0;
    data[1] = 1;
    data[2] = 0;
    data[3] = 0;
    data[4] = 1;

    Vector stencil(5);
    stencil[0] = 1;
    stencil[1] = 2;
    stencil[2] = 1;
    stencil[3] = 1;
    stencil[4] = 2;

    Vector true_results(2);
    Vector false_results(3);

    thrust::pair<typename Vector::iterator, typename Vector::iterator> ends
        = thrust::partition_copy(data.begin(),
                                 data.end(),
                                 stencil.begin(),
                                 true_results.begin(),
                                 false_results.begin(),
                                 is_even<T>());

    Vector true_ref(2);
    true_ref[0] = 1;
    true_ref[1] = 1;

    Vector false_ref(3);
    false_ref[0] = 0;
    false_ref[1] = 0;
    false_ref[2] = 0;

    ASSERT_EQ(2, ends.first - true_results.begin());
    ASSERT_EQ(3, ends.second - false_results.begin());
    ASSERT_EQ(true_ref, true_results);
    ASSERT_EQ(false_ref, false_results);
}

TYPED_TEST(PartitionVectorTests, TestStablePartitionSimple)
{
    using Vector   = typename TestFixture::input_type;
    using T        = typename Vector::value_type;
    using Iterator = typename Vector::iterator;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector data(5);
    data[0] = 1;
    data[1] = 2;
    data[2] = 1;
    data[3] = 3;
    data[4] = 2;

    Iterator iter = thrust::stable_partition(data.begin(), data.end(), is_even<T>());

    Vector ref(5);
    ref[0] = 2;
    ref[1] = 2;
    ref[2] = 1;
    ref[3] = 1;
    ref[4] = 3;

    ASSERT_EQ(iter - data.begin(), 2);
    ASSERT_EQ(data, ref);
}

TYPED_TEST(PartitionVectorTests, TestStablePartitionStencilSimple)
{
    using Vector   = typename TestFixture::input_type;
    using T        = typename Vector::value_type;
    using Iterator = typename Vector::iterator;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector data(5);
    data[0] = 1;
    data[1] = 2;
    data[2] = 1;
    data[3] = 3;
    data[4] = 2;

    Vector stencil(5);
    stencil[0] = 0;
    stencil[1] = 1;
    stencil[2] = 0;
    stencil[3] = 0;
    stencil[4] = 1;

    Iterator iter = thrust::stable_partition(
        data.begin(), data.end(), stencil.begin(), thrust::identity<T>());

    Vector ref(5);
    ref[0] = 2;
    ref[1] = 2;
    ref[2] = 1;
    ref[3] = 1;
    ref[4] = 3;

    ASSERT_EQ(iter - data.begin(), 2);
    ASSERT_EQ(data, ref);
}

TYPED_TEST(PartitionVectorTests, TestStablePartitionCopySimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector data(5);
    data[0] = 1;
    data[1] = 2;
    data[2] = 1;
    data[3] = 1;
    data[4] = 2;

    Vector true_results(2);
    Vector false_results(3);

    thrust::pair<typename Vector::iterator, typename Vector::iterator> ends
        = thrust::stable_partition_copy(
            data.begin(), data.end(), true_results.begin(), false_results.begin(), is_even<T>());

    Vector true_ref(2);
    true_ref[0] = 2;
    true_ref[1] = 2;

    Vector false_ref(3);
    false_ref[0] = 1;
    false_ref[1] = 1;
    false_ref[2] = 1;

    ASSERT_EQ(2, ends.first - true_results.begin());
    ASSERT_EQ(3, ends.second - false_results.begin());
    ASSERT_EQ(true_ref, true_results);
    ASSERT_EQ(false_ref, false_results);
}

TYPED_TEST(PartitionVectorTests, TestStablePartitionCopyStencilSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector data(5);
    data[0] = 1;
    data[1] = 2;
    data[2] = 1;
    data[3] = 1;
    data[4] = 2;

    Vector stencil(5);
    stencil[0] = false;
    stencil[1] = true;
    stencil[2] = false;
    stencil[3] = false;
    stencil[4] = true;

    Vector true_results(2);
    Vector false_results(3);

    thrust::pair<typename Vector::iterator, typename Vector::iterator> ends
        = thrust::stable_partition_copy(data.begin(),
                                        data.end(),
                                        stencil.begin(),
                                        true_results.begin(),
                                        false_results.begin(),
                                        thrust::identity<T>());

    Vector true_ref(2);
    true_ref[0] = 2;
    true_ref[1] = 2;

    Vector false_ref(3);
    false_ref[0] = 1;
    false_ref[1] = 1;
    false_ref[2] = 1;

    ASSERT_EQ(2, ends.first - true_results.begin());
    ASSERT_EQ(3, ends.second - false_results.begin());
    ASSERT_EQ(true_ref, true_results);
    ASSERT_EQ(false_ref, false_results);
}

TYPED_TEST(PartitionIntegerTests, TestPartition)
{
    using T                         = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            // setup ranges
            thrust::host_vector<T> h_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
            thrust::device_vector<T> d_data = h_data;

            typename thrust::host_vector<T>::iterator h_iter
                = thrust::partition(h_data.begin(), h_data.end(), is_even<T>());
            typename thrust::device_vector<T>::iterator d_iter
                = thrust::partition(d_data.begin(), d_data.end(), is_even<T>());

            size_t h_iter_size1 = static_cast<size_t>(thrust::distance(h_data.begin(), h_iter));
            size_t d_iter_size1 = static_cast<size_t>(thrust::distance(d_data.begin(), d_iter));
            size_t h_iter_size2 = static_cast<size_t>(thrust::distance(h_iter, h_data.end()));
            size_t d_iter_size2 = static_cast<size_t>(thrust::distance(d_iter, d_data.end()));

            // TODO: Temporary workaround for size 0 inputs with grid/block min dim > 1
            if (h_iter_size1 == 0 || d_iter_size1 == 0 || h_iter_size2 == 0 || d_iter_size2 == 0)
            {
                continue;
            }

            thrust::sort(h_data.begin(), h_iter);
            thrust::sort(h_iter, h_data.end());
            thrust::sort(d_data.begin(), d_iter);
            thrust::sort(d_iter, d_data.end());

            ASSERT_EQ(h_data, d_data);
            ASSERT_EQ(h_iter - h_data.begin(), d_iter - d_data.begin());
        }
    }
};

TYPED_TEST(PartitionIntegerTests, TestPartitionStencil)
{
    using T                         = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            // setup ranges
            thrust::host_vector<T> h_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
            thrust::host_vector<T> h_stencil = get_random_data<T>(
                size,
                std::numeric_limits<T>::min(),
                std::numeric_limits<T>::max(),
                seed + seed_value_addition);
            thrust::device_vector<T> d_data    = h_data;
            thrust::device_vector<T> d_stencil = h_stencil;

            typename thrust::host_vector<T>::iterator h_iter
                = thrust::partition(h_data.begin(), h_data.end(), h_stencil.begin(), is_even<T>());
            typename thrust::device_vector<T>::iterator d_iter
                = thrust::partition(d_data.begin(), d_data.end(), d_stencil.begin(), is_even<T>());

            size_t h_iter_size1 = static_cast<size_t>(thrust::distance(h_data.begin(), h_iter));
            size_t d_iter_size1 = static_cast<size_t>(thrust::distance(d_data.begin(), d_iter));
            size_t h_iter_size2 = static_cast<size_t>(thrust::distance(h_iter, h_data.end()));
            size_t d_iter_size2 = static_cast<size_t>(thrust::distance(d_iter, d_data.end()));

            // TODO: Temporary workaround for size 0 inputs with grid/block min dim > 1
            if (h_iter_size1 == 0 || d_iter_size1 == 0 || h_iter_size2 == 0 || d_iter_size2 == 0)
            {
                continue;
            }

            thrust::sort(h_data.begin(), h_iter);
            thrust::sort(h_iter, h_data.end());
            thrust::sort(d_data.begin(), d_iter);
            thrust::sort(d_iter, d_data.end());

            ASSERT_EQ(h_data, d_data);
            ASSERT_EQ(h_iter - h_data.begin(), d_iter - d_data.begin());
        }
    }
};

TYPED_TEST(PartitionIntegerTests, TestPartitionCopy)
{
    using T                         = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            // setup input ranges
            thrust::host_vector<T> h_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
            thrust::device_vector<T> d_data = h_data;

            size_t n_true  = thrust::count_if(h_data.begin(), h_data.end(), is_even<T>());
            size_t n_false = size - n_true;

            // TODO: Temporary workaround for size 0 inputs with grid/block min dim > 1
            if (n_true == 0 || n_false == 0) continue;

            // setup output ranges
            thrust::host_vector<T>   h_true_results(n_true, 0);
            thrust::host_vector<T>   h_false_results(n_false, 0);
            thrust::device_vector<T> d_true_results(n_true, 0);
            thrust::device_vector<T> d_false_results(n_false, 0);

            thrust::pair<typename thrust::host_vector<T>::iterator,
                         typename thrust::host_vector<T>::iterator>
                h_ends = thrust::partition_copy(h_data.begin(),
                                                h_data.end(),
                                                h_true_results.begin(),
                                                h_false_results.begin(),
                                                is_even<T>());

            thrust::pair<typename thrust::device_vector<T>::iterator,
                         typename thrust::device_vector<T>::iterator>
                d_ends = thrust::partition_copy(d_data.begin(),
                                                d_data.end(),
                                                d_true_results.begin(),
                                                d_false_results.begin(),
                                                is_even<T>());

            // check true output
            ASSERT_EQ(h_ends.first - h_true_results.begin(), n_true);
            ASSERT_EQ(d_ends.first - d_true_results.begin(), n_true);
            thrust::sort(h_true_results.begin(), h_true_results.end());
            thrust::sort(d_true_results.begin(), d_true_results.end());
            ASSERT_EQ(h_true_results, d_true_results);

            // check false output
            ASSERT_EQ(h_ends.second - h_false_results.begin(), n_false);
            ASSERT_EQ(d_ends.second - d_false_results.begin(), n_false);
            thrust::sort(h_false_results.begin(), h_false_results.end());
            thrust::sort(d_false_results.begin(), d_false_results.end());
            ASSERT_EQ(h_false_results, d_false_results);
        }
    }
};

TYPED_TEST(PartitionIntegerTests, TestPartitionCopyStencil)
{
    using T                         = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            // setup input ranges
            thrust::host_vector<T> h_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
            thrust::host_vector<T> h_stencil = get_random_data<T>(
                size,
                std::numeric_limits<T>::min(),
                std::numeric_limits<T>::max(),
                seed + seed_value_addition
            );
            thrust::device_vector<T> d_data    = h_data;
            thrust::device_vector<T> d_stencil = h_stencil;

            size_t n_true  = thrust::count_if(h_stencil.begin(), h_stencil.end(), is_even<T>());
            size_t n_false = size - n_true;

            // TODO: Temporary workaround for size 0 inputs with grid/block min dim > 1
            if (n_true == 0 || n_false == 0) continue;

            // setup output ranges
            thrust::host_vector<T>   h_true_results(n_true, 0);
            thrust::host_vector<T>   h_false_results(n_false, 0);
            thrust::device_vector<T> d_true_results(n_true, 0);
            thrust::device_vector<T> d_false_results(n_false, 0);

            thrust::pair<typename thrust::host_vector<T>::iterator,
                         typename thrust::host_vector<T>::iterator>
                h_ends = thrust::partition_copy(h_data.begin(),
                                                h_data.end(),
                                                h_stencil.begin(),
                                                h_true_results.begin(),
                                                h_false_results.begin(),
                                                is_even<T>());

            thrust::pair<typename thrust::device_vector<T>::iterator,
                         typename thrust::device_vector<T>::iterator>
                d_ends = thrust::partition_copy(d_data.begin(),
                                                d_data.end(),
                                                d_stencil.begin(),
                                                d_true_results.begin(),
                                                d_false_results.begin(),
                                                is_even<T>());

            // check true output
            ASSERT_EQ(h_ends.first - h_true_results.begin(), n_true);
            ASSERT_EQ(d_ends.first - d_true_results.begin(), n_true);
            thrust::sort(h_true_results.begin(), h_true_results.end());
            thrust::sort(d_true_results.begin(), d_true_results.end());
            ASSERT_EQ(h_true_results, d_true_results);

            // check false output
            ASSERT_EQ(h_ends.second - h_false_results.begin(), n_false);
            ASSERT_EQ(d_ends.second - d_false_results.begin(), n_false);
            thrust::sort(h_false_results.begin(), h_false_results.end());
            thrust::sort(d_false_results.begin(), d_false_results.end());
            ASSERT_EQ(h_false_results, d_false_results);
        }
    }
};

TYPED_TEST(PartitionIntegerTests, TestStablePartitionCopyStencil)
{
    using T                         = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            // setup input ranges
            thrust::host_vector<T> h_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
            thrust::host_vector<T> h_stencil = get_random_data<T>(
                size,
                std::numeric_limits<T>::min(),
                std::numeric_limits<T>::max(),
                seed + seed_value_addition
            );
            thrust::device_vector<T> d_data    = h_data;
            thrust::device_vector<T> d_stencil = h_stencil;

            size_t n_true  = thrust::count_if(h_stencil.begin(), h_stencil.end(), is_even<T>());
            size_t n_false = size - n_true;

            // TODO: Temporary workaround for size 0 inputs with grid/block min dim > 1
            if (n_true == 0 || n_false == 0) continue;

            // setup output ranges
            thrust::host_vector<T>   h_true_results(n_true, 0);
            thrust::host_vector<T>   h_false_results(n_false, 0);
            thrust::device_vector<T> d_true_results(n_true, 0);
            thrust::device_vector<T> d_false_results(n_false, 0);

            thrust::pair<typename thrust::host_vector<T>::iterator,
                         typename thrust::host_vector<T>::iterator>
                h_ends = thrust::stable_partition_copy(h_data.begin(),
                                                       h_data.end(),
                                                       h_stencil.begin(),
                                                       h_true_results.begin(),
                                                       h_false_results.begin(),
                                                       is_even<T>());

            thrust::pair<typename thrust::device_vector<T>::iterator,
                         typename thrust::device_vector<T>::iterator>
                d_ends = thrust::stable_partition_copy(d_data.begin(),
                                                       d_data.end(),
                                                       d_stencil.begin(),
                                                       d_true_results.begin(),
                                                       d_false_results.begin(),
                                                       is_even<T>());

            // check true output
            ASSERT_EQ(h_ends.first - h_true_results.begin(), n_true);
            ASSERT_EQ(d_ends.first - d_true_results.begin(), n_true);
            thrust::sort(h_true_results.begin(), h_true_results.end());
            thrust::sort(d_true_results.begin(), d_true_results.end());
            ASSERT_EQ(h_true_results, d_true_results);

            // check false output
            ASSERT_EQ(h_ends.second - h_false_results.begin(), n_false);
            ASSERT_EQ(d_ends.second - d_false_results.begin(), n_false);
            thrust::sort(h_false_results.begin(), h_false_results.end());
            thrust::sort(d_false_results.begin(), d_false_results.end());
            ASSERT_EQ(h_false_results, d_false_results);
        }
    }
};

TYPED_TEST(PartitionIntegerTests, TestPartitionCopyToDiscardIterator)
{
    using T                         = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            // setup input ranges
            thrust::host_vector<T> h_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
            thrust::device_vector<T> d_data = h_data;

            size_t n_true  = thrust::count_if(h_data.begin(), h_data.end(), is_even<T>());
            size_t n_false = size - n_true;

            // TODO: Temporary workaround for size 0 inputs with grid/block min dim > 1
            if (n_true == 0 || n_false == 0) continue;

            // mask both ranges
            thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> h_result1
                = thrust::partition_copy(h_data.begin(),
                                         h_data.end(),
                                         thrust::make_discard_iterator(),
                                         thrust::make_discard_iterator(),
                                         is_even<T>());

            thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> d_result1
                = thrust::partition_copy(d_data.begin(),
                                         d_data.end(),
                                         thrust::make_discard_iterator(),
                                         thrust::make_discard_iterator(),
                                         is_even<T>());

            thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> reference1
                = thrust::make_pair(thrust::make_discard_iterator(n_true),
                                    thrust::make_discard_iterator(n_false));

            ASSERT_EQ_QUIET(reference1, h_result1);
            ASSERT_EQ_QUIET(reference1, d_result1);

            // mask the false range
            thrust::host_vector<T>   h_trues(n_true);
            thrust::device_vector<T> d_trues(n_true);

            thrust::pair<typename thrust::host_vector<T>::iterator, thrust::discard_iterator<>>
                h_result2 = thrust::partition_copy(h_data.begin(),
                                                   h_data.end(),
                                                   h_trues.begin(),
                                                   thrust::make_discard_iterator(),
                                                   is_even<T>());

            thrust::pair<typename thrust::device_vector<T>::iterator, thrust::discard_iterator<>>
                d_result2 = thrust::partition_copy(d_data.begin(),
                                                   d_data.end(),
                                                   d_trues.begin(),
                                                   thrust::make_discard_iterator(),
                                                   is_even<T>());

            thrust::pair<typename thrust::host_vector<T>::iterator, thrust::discard_iterator<>>
                h_reference2 = thrust::make_pair(h_trues.begin() + n_true,
                                                 thrust::make_discard_iterator(n_false));

            thrust::pair<typename thrust::device_vector<T>::iterator, thrust::discard_iterator<>>
                d_reference2 = thrust::make_pair(d_trues.begin() + n_true,
                                                 thrust::make_discard_iterator(n_false));

            ASSERT_EQ(h_trues, d_trues);
            ASSERT_EQ_QUIET(h_reference2, h_result2);
            ASSERT_EQ_QUIET(d_reference2, d_result2);

            // mask the true range
            thrust::host_vector<T>   h_falses(n_false);
            thrust::device_vector<T> d_falses(n_false);

            thrust::pair<thrust::discard_iterator<>, typename thrust::host_vector<T>::iterator>
                h_result3 = thrust::partition_copy(h_data.begin(),
                                                   h_data.end(),
                                                   thrust::make_discard_iterator(),
                                                   h_falses.begin(),
                                                   is_even<T>());

            thrust::pair<thrust::discard_iterator<>, typename thrust::device_vector<T>::iterator>
                d_result3 = thrust::partition_copy(d_data.begin(),
                                                   d_data.end(),
                                                   thrust::make_discard_iterator(),
                                                   d_falses.begin(),
                                                   is_even<T>());

            thrust::pair<thrust::discard_iterator<>, typename thrust::host_vector<T>::iterator>
                h_reference3 = thrust::make_pair(thrust::make_discard_iterator(n_true),
                                                 h_falses.begin() + n_false);

            thrust::pair<thrust::discard_iterator<>, typename thrust::device_vector<T>::iterator>
                d_reference3 = thrust::make_pair(thrust::make_discard_iterator(n_true),
                                                 d_falses.begin() + n_false);

            ASSERT_EQ(h_falses, d_falses);
            ASSERT_EQ_QUIET(h_reference3, h_result3);
            ASSERT_EQ_QUIET(d_reference3, d_result3);
        }
    }
};

TYPED_TEST(PartitionIntegerTests, TestPartitionCopyStencilToDiscardIterator)
{
    using T                         = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            // setup input ranges
            thrust::host_vector<T> h_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
            thrust::host_vector<T> h_stencil = get_random_data<T>(
                size,
                std::numeric_limits<T>::min(),
                std::numeric_limits<T>::max(),
                seed + seed_value_addition
            );

            thrust::device_vector<T> d_data    = h_data;
            thrust::device_vector<T> d_stencil = h_stencil;

            size_t n_true  = thrust::count_if(h_stencil.begin(), h_stencil.end(), is_even<T>());
            size_t n_false = size - n_true;

            // TODO: Temporary workaround for size 0 inputs with grid/block min dim > 1
            if (n_true == 0 || n_false == 0) continue;

            // mask both ranges
            thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> h_result1
                = thrust::partition_copy(h_data.begin(),
                                         h_data.end(),
                                         h_stencil.begin(),
                                         thrust::make_discard_iterator(),
                                         thrust::make_discard_iterator(),
                                         is_even<T>());

            thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> d_result1
                = thrust::partition_copy(d_data.begin(),
                                         d_data.end(),
                                         d_stencil.begin(),
                                         thrust::make_discard_iterator(),
                                         thrust::make_discard_iterator(),
                                         is_even<T>());

            thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> reference1
                = thrust::make_pair(thrust::make_discard_iterator(n_true),
                                    thrust::make_discard_iterator(n_false));

            ASSERT_EQ_QUIET(reference1, h_result1);
            ASSERT_EQ_QUIET(reference1, d_result1);

            // mask the false range
            thrust::host_vector<T>   h_trues(n_true);
            thrust::device_vector<T> d_trues(n_true);

            thrust::pair<typename thrust::host_vector<T>::iterator, thrust::discard_iterator<>>
                h_result2 = thrust::partition_copy(h_data.begin(),
                                                   h_data.end(),
                                                   h_stencil.begin(),
                                                   h_trues.begin(),
                                                   thrust::make_discard_iterator(),
                                                   is_even<T>());

            thrust::pair<typename thrust::device_vector<T>::iterator, thrust::discard_iterator<>>
                d_result2 = thrust::partition_copy(d_data.begin(),
                                                   d_data.end(),
                                                   d_stencil.begin(),
                                                   d_trues.begin(),
                                                   thrust::make_discard_iterator(),
                                                   is_even<T>());

            thrust::pair<typename thrust::host_vector<T>::iterator, thrust::discard_iterator<>>
                h_reference2 = thrust::make_pair(h_trues.begin() + n_true,
                                                 thrust::make_discard_iterator(n_false));

            thrust::pair<typename thrust::device_vector<T>::iterator, thrust::discard_iterator<>>
                d_reference2 = thrust::make_pair(d_trues.begin() + n_true,
                                                 thrust::make_discard_iterator(n_false));

            ASSERT_EQ(h_trues, d_trues);
            ASSERT_EQ_QUIET(h_reference2, h_result2);
            ASSERT_EQ_QUIET(d_reference2, d_result2);

            // mask the true range
            thrust::host_vector<T>   h_falses(n_false);
            thrust::device_vector<T> d_falses(n_false);

            thrust::pair<thrust::discard_iterator<>, typename thrust::host_vector<T>::iterator>
                h_result3 = thrust::partition_copy(h_data.begin(),
                                                   h_data.end(),
                                                   h_stencil.begin(),
                                                   thrust::make_discard_iterator(),
                                                   h_falses.begin(),
                                                   is_even<T>());

            thrust::pair<thrust::discard_iterator<>, typename thrust::device_vector<T>::iterator>
                d_result3 = thrust::partition_copy(d_data.begin(),
                                                   d_data.end(),
                                                   d_stencil.begin(),
                                                   thrust::make_discard_iterator(),
                                                   d_falses.begin(),
                                                   is_even<T>());

            thrust::pair<thrust::discard_iterator<>, typename thrust::host_vector<T>::iterator>
                h_reference3 = thrust::make_pair(thrust::make_discard_iterator(n_true),
                                                 h_falses.begin() + n_false);

            thrust::pair<thrust::discard_iterator<>, typename thrust::device_vector<T>::iterator>
                d_reference3 = thrust::make_pair(thrust::make_discard_iterator(n_true),
                                                 d_falses.begin() + n_false);

            ASSERT_EQ(h_falses, d_falses);
            ASSERT_EQ_QUIET(h_reference3, h_result3);
            ASSERT_EQ_QUIET(d_reference3, d_result3);
        }
    }
};

TYPED_TEST(PartitionIntegerTests, TestStablePartition)
{
    using T                         = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            // setup ranges
            thrust::host_vector<T> h_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
            thrust::device_vector<T> d_data = h_data;

            typename thrust::host_vector<T>::iterator h_iter
                = thrust::stable_partition(h_data.begin(), h_data.end(), is_even<T>());
            typename thrust::device_vector<T>::iterator d_iter
                = thrust::stable_partition(d_data.begin(), d_data.end(), is_even<T>());

            ASSERT_EQ(h_data, d_data);
            ASSERT_EQ(h_iter - h_data.begin(), d_iter - d_data.begin());
        }
    }
};

TYPED_TEST(PartitionIntegerTests, TestStablePartitionStencil)
{
    using T                         = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            // setup ranges
            thrust::host_vector<T> h_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
            thrust::host_vector<T> h_stencil = get_random_data<T>(
                size,
                std::numeric_limits<T>::min(),
                std::numeric_limits<T>::max(),
                seed + seed_value_addition
            );
            thrust::device_vector<T> d_data    = h_data;
            thrust::device_vector<T> d_stencil = h_stencil;

            typename thrust::host_vector<T>::iterator h_iter = thrust::stable_partition(
                h_data.begin(), h_data.end(), h_stencil.begin(), is_even<T>());
            typename thrust::device_vector<T>::iterator d_iter = thrust::stable_partition(
                d_data.begin(), d_data.end(), d_stencil.begin(), is_even<T>());

            ASSERT_EQ(h_data, d_data);
            ASSERT_EQ(h_iter - h_data.begin(), d_iter - d_data.begin());
        }
    }
};

TYPED_TEST(PartitionIntegerTests, TestStablePartitionCopy)
{
    using T                         = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            // setup input ranges
            thrust::host_vector<T> h_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
            thrust::device_vector<T> d_data = h_data;

            size_t n_true  = thrust::count_if(h_data.begin(), h_data.end(), is_even<T>());
            size_t n_false = size - n_true;

            // TODO: Temporary workaround for size 0 inputs with grid/block min dim > 1
            if (n_true == 0 || n_false == 0) continue;

            // setup output ranges
            thrust::host_vector<T>   h_true_results(n_true, 0);
            thrust::host_vector<T>   h_false_results(n_false, 0);
            thrust::device_vector<T> d_true_results(n_true, 0);
            thrust::device_vector<T> d_false_results(n_false, 0);

            thrust::pair<typename thrust::host_vector<T>::iterator,
                         typename thrust::host_vector<T>::iterator>
                h_ends = thrust::stable_partition_copy(h_data.begin(),
                                                       h_data.end(),
                                                       h_true_results.begin(),
                                                       h_false_results.begin(),
                                                       is_even<T>());

            thrust::pair<typename thrust::device_vector<T>::iterator,
                         typename thrust::device_vector<T>::iterator>
                d_ends = thrust::stable_partition_copy(d_data.begin(),
                                                       d_data.end(),
                                                       d_true_results.begin(),
                                                       d_false_results.begin(),
                                                       is_even<T>());

            // check true output
            ASSERT_EQ(h_ends.first - h_true_results.begin(), n_true);
            ASSERT_EQ(d_ends.first - d_true_results.begin(), n_true);
            ASSERT_EQ(h_true_results, d_true_results);

            // check false output
            ASSERT_EQ(h_ends.second - h_false_results.begin(), n_false);
            ASSERT_EQ(d_ends.second - d_false_results.begin(), n_false);
            ASSERT_EQ(h_false_results, d_false_results);
        }
    }
};

TYPED_TEST(PartitionIntegerTests, TestStablePartitionCopyToDiscardIterator)
{
    using T                         = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            // setup input ranges
            thrust::host_vector<T> h_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
            thrust::device_vector<T> d_data = h_data;

            size_t n_true  = thrust::count_if(h_data.begin(), h_data.end(), is_even<T>());
            size_t n_false = size - n_true;

            // TODO: Temporary workaround for size 0 inputs with grid/block min dim > 1
            if (n_true == 0 || n_false == 0) continue;

            // mask both ranges
            thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> h_result1
                = thrust::stable_partition_copy(h_data.begin(),
                                                h_data.end(),
                                                thrust::make_discard_iterator(),
                                                thrust::make_discard_iterator(),
                                                is_even<T>());

            thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> d_result1
                = thrust::stable_partition_copy(d_data.begin(),
                                                d_data.end(),
                                                thrust::make_discard_iterator(),
                                                thrust::make_discard_iterator(),
                                                is_even<T>());

            thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> reference1
                = thrust::make_pair(thrust::make_discard_iterator(n_true),
                                    thrust::make_discard_iterator(n_false));

            ASSERT_EQ_QUIET(reference1, h_result1);
            ASSERT_EQ_QUIET(reference1, d_result1);

            // mask the false range
            thrust::host_vector<T>   h_trues(n_true);
            thrust::device_vector<T> d_trues(n_true);

            thrust::pair<typename thrust::host_vector<T>::iterator, thrust::discard_iterator<>>
                h_result2 = thrust::stable_partition_copy(h_data.begin(),
                                                          h_data.end(),
                                                          h_trues.begin(),
                                                          thrust::make_discard_iterator(),
                                                          is_even<T>());

            thrust::pair<typename thrust::device_vector<T>::iterator, thrust::discard_iterator<>>
                d_result2 = thrust::stable_partition_copy(d_data.begin(),
                                                          d_data.end(),
                                                          d_trues.begin(),
                                                          thrust::make_discard_iterator(),
                                                          is_even<T>());

            thrust::pair<typename thrust::host_vector<T>::iterator, thrust::discard_iterator<>>
                h_reference2 = thrust::make_pair(h_trues.begin() + n_true,
                                                 thrust::make_discard_iterator(n_false));

            thrust::pair<typename thrust::device_vector<T>::iterator, thrust::discard_iterator<>>
                d_reference2 = thrust::make_pair(d_trues.begin() + n_true,
                                                 thrust::make_discard_iterator(n_false));

            ASSERT_EQ(h_trues, d_trues);
            ASSERT_EQ_QUIET(h_reference2, h_result2);
            ASSERT_EQ_QUIET(d_reference2, d_result2);

            // mask the true range
            thrust::host_vector<T>   h_falses(n_false);
            thrust::device_vector<T> d_falses(n_false);

            thrust::pair<thrust::discard_iterator<>, typename thrust::host_vector<T>::iterator>
                h_result3 = thrust::stable_partition_copy(h_data.begin(),
                                                          h_data.end(),
                                                          thrust::make_discard_iterator(),
                                                          h_falses.begin(),
                                                          is_even<T>());

            thrust::pair<thrust::discard_iterator<>, typename thrust::device_vector<T>::iterator>
                d_result3 = thrust::stable_partition_copy(d_data.begin(),
                                                          d_data.end(),
                                                          thrust::make_discard_iterator(),
                                                          d_falses.begin(),
                                                          is_even<T>());

            thrust::pair<thrust::discard_iterator<>, typename thrust::host_vector<T>::iterator>
                h_reference3 = thrust::make_pair(thrust::make_discard_iterator(n_true),
                                                 h_falses.begin() + n_false);

            thrust::pair<thrust::discard_iterator<>, typename thrust::device_vector<T>::iterator>
                d_reference3 = thrust::make_pair(thrust::make_discard_iterator(n_true),
                                                 d_falses.begin() + n_false);

            ASSERT_EQ(h_falses, d_falses);
            ASSERT_EQ_QUIET(h_reference3, h_result3);
            ASSERT_EQ_QUIET(d_reference3, d_result3);
        }
    }
};

TYPED_TEST(PartitionIntegerTests, TestStablePartitionCopyStencilToDiscardIterator)
{
    using T                         = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            // setup input ranges
            thrust::host_vector<T> h_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed);
            thrust::host_vector<T> h_stencil = get_random_data<T>(
                size,
                std::numeric_limits<T>::min(),
                std::numeric_limits<T>::max(),
                seed + seed_value_addition
            );
            thrust::device_vector<T> d_data    = h_data;
            thrust::device_vector<T> d_stencil = h_stencil;

            size_t n_true  = thrust::count_if(h_stencil.begin(), h_stencil.end(), is_even<T>());
            size_t n_false = size - n_true;

            // TODO: Temporary workaround for size 0 inputs with grid/block min dim > 1
            if (n_true == 0 || n_false == 0) continue;

            // mask both ranges
            thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> h_result1
                = thrust::stable_partition_copy(h_data.begin(),
                                                h_data.end(),
                                                h_stencil.begin(),
                                                thrust::make_discard_iterator(),
                                                thrust::make_discard_iterator(),
                                                is_even<T>());

            thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> d_result1
                = thrust::stable_partition_copy(d_data.begin(),
                                                d_data.end(),
                                                d_stencil.begin(),
                                                thrust::make_discard_iterator(),
                                                thrust::make_discard_iterator(),
                                                is_even<T>());

            thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> reference1
                = thrust::make_pair(thrust::make_discard_iterator(n_true),
                                    thrust::make_discard_iterator(n_false));

            ASSERT_EQ_QUIET(reference1, h_result1);
            ASSERT_EQ_QUIET(reference1, d_result1);

            // mask the false range
            thrust::host_vector<T>   h_trues(n_true);
            thrust::device_vector<T> d_trues(n_true);

            thrust::pair<typename thrust::host_vector<T>::iterator, thrust::discard_iterator<>>
                h_result2 = thrust::stable_partition_copy(h_data.begin(),
                                                          h_data.end(),
                                                          h_stencil.begin(),
                                                          h_trues.begin(),
                                                          thrust::make_discard_iterator(),
                                                          is_even<T>());

            thrust::pair<typename thrust::device_vector<T>::iterator, thrust::discard_iterator<>>
                d_result2 = thrust::stable_partition_copy(d_data.begin(),
                                                          d_data.end(),
                                                          d_stencil.begin(),
                                                          d_trues.begin(),
                                                          thrust::make_discard_iterator(),
                                                          is_even<T>());

            thrust::pair<typename thrust::host_vector<T>::iterator, thrust::discard_iterator<>>
                h_reference2 = thrust::make_pair(h_trues.begin() + n_true,
                                                 thrust::make_discard_iterator(n_false));

            thrust::pair<typename thrust::device_vector<T>::iterator, thrust::discard_iterator<>>
                d_reference2 = thrust::make_pair(d_trues.begin() + n_true,
                                                 thrust::make_discard_iterator(n_false));

            ASSERT_EQ(h_trues, d_trues);
            ASSERT_EQ_QUIET(h_reference2, h_result2);
            ASSERT_EQ_QUIET(d_reference2, d_result2);

            // mask the true range
            thrust::host_vector<T>   h_falses(n_false);
            thrust::device_vector<T> d_falses(n_false);

            thrust::pair<thrust::discard_iterator<>, typename thrust::host_vector<T>::iterator>
                h_result3 = thrust::stable_partition_copy(h_data.begin(),
                                                          h_data.end(),
                                                          h_stencil.begin(),
                                                          thrust::make_discard_iterator(),
                                                          h_falses.begin(),
                                                          is_even<T>());

            thrust::pair<thrust::discard_iterator<>, typename thrust::device_vector<T>::iterator>
                d_result3 = thrust::stable_partition_copy(d_data.begin(),
                                                          d_data.end(),
                                                          d_stencil.begin(),
                                                          thrust::make_discard_iterator(),
                                                          d_falses.begin(),
                                                          is_even<T>());

            thrust::pair<thrust::discard_iterator<>, typename thrust::host_vector<T>::iterator>
                h_reference3 = thrust::make_pair(thrust::make_discard_iterator(n_true),
                                                 h_falses.begin() + n_false);

            thrust::pair<thrust::discard_iterator<>, typename thrust::device_vector<T>::iterator>
                d_reference3 = thrust::make_pair(thrust::make_discard_iterator(n_true),
                                                 d_falses.begin() + n_false);

            ASSERT_EQ(h_falses, d_falses);
            ASSERT_EQ_QUIET(h_reference3, h_result3);
            ASSERT_EQ_QUIET(d_reference3, d_result3);
        }
    }
};

struct is_ordered
{
    template <typename Tuple>
    __host__ __device__ bool operator()(const Tuple& t) const
    {
        return thrust::get<0>(t) <= thrust::get<1>(t);
    }
};

TYPED_TEST(PartitionVectorTests, TestPartitionZipIterator)
{
    using Vector = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector data1(5);
    Vector data2(5);

    data1[0] = 1;
    data2[0] = 2;
    data1[1] = 2;
    data2[1] = 1;
    data1[2] = 1;
    data2[2] = 2;
    data1[3] = 1;
    data2[3] = 2;
    data1[4] = 2;
    data2[4] = 1;

    using Iterator      = typename Vector::iterator;
    using IteratorTuple = typename thrust::tuple<Iterator, Iterator>;
    using ZipIterator   = typename thrust::zip_iterator<IteratorTuple>;

    ZipIterator begin = thrust::make_zip_iterator(thrust::make_tuple(data1.begin(), data2.begin()));
    ZipIterator end   = thrust::make_zip_iterator(thrust::make_tuple(data1.end(), data2.end()));

    ZipIterator iter = thrust::partition(begin, end, is_ordered());

    Vector ref1(5);
    Vector ref2(5);

    ref1[0] = 1;
    ref2[0] = 2;
    ref1[1] = 1;
    ref2[1] = 2;
    ref1[2] = 1;
    ref2[2] = 2;
    ref1[3] = 2;
    ref2[3] = 1;
    ref1[4] = 2;
    ref2[4] = 1;

    ASSERT_EQ(iter - begin, 3);
    ASSERT_EQ(data1, ref1);
    ASSERT_EQ(data2, ref2);
}

TYPED_TEST(PartitionVectorTests, TestPartitionStencilZipIterator)
{
    using Vector = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector data(5);
    data[0] = 1;
    data[1] = 0;
    data[2] = 1;
    data[3] = 1;
    data[4] = 0;

    Vector stencil1(5);
    Vector stencil2(5);

    stencil1[0] = 1;
    stencil2[0] = 2;
    stencil1[1] = 2;
    stencil2[1] = 1;
    stencil1[2] = 1;
    stencil2[2] = 2;
    stencil1[3] = 1;
    stencil2[3] = 2;
    stencil1[4] = 2;
    stencil2[4] = 1;

    using Iterator      = typename Vector::iterator;
    using IteratorTuple = typename thrust::tuple<Iterator, Iterator>;
    using ZipIterator   = typename thrust::zip_iterator<IteratorTuple>;

    ZipIterator stencil_begin
        = thrust::make_zip_iterator(thrust::make_tuple(stencil1.begin(), stencil2.begin()));

    Iterator iter = thrust::partition(data.begin(), data.end(), stencil_begin, is_ordered());

    Vector ref(5);

    ref[0] = 1;
    ref[1] = 1;
    ref[2] = 1;
    ref[3] = 0;
    ref[4] = 0;

    ASSERT_EQ(iter - data.begin(), 3);
    ASSERT_EQ(data, ref);
}

TYPED_TEST(PartitionVectorTests, TestStablePartitionZipIterator)
{
    using Vector = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector data1(5);
    Vector data2(5);

    data1[0] = 1;
    data2[0] = 2;
    data1[1] = 2;
    data2[1] = 0;
    data1[2] = 1;
    data2[2] = 3;
    data1[3] = 1;
    data2[3] = 2;
    data1[4] = 2;
    data2[4] = 1;

    using Iterator      = typename Vector::iterator;
    using IteratorTuple = typename thrust::tuple<Iterator, Iterator>;
    using ZipIterator   = typename thrust::zip_iterator<IteratorTuple>;

    ZipIterator begin = thrust::make_zip_iterator(thrust::make_tuple(data1.begin(), data2.begin()));
    ZipIterator end   = thrust::make_zip_iterator(thrust::make_tuple(data1.end(), data2.end()));

    ZipIterator iter = thrust::stable_partition(begin, end, is_ordered());

    Vector ref1(5);
    Vector ref2(5);

    ref1[0] = 1;
    ref2[0] = 2;
    ref1[1] = 1;
    ref2[1] = 3;
    ref1[2] = 1;
    ref2[2] = 2;
    ref1[3] = 2;
    ref2[3] = 0;
    ref1[4] = 2;
    ref2[4] = 1;

    ASSERT_EQ(data1, ref1);
    ASSERT_EQ(data2, ref2);
    ASSERT_EQ(iter - begin, 3);
}

TYPED_TEST(PartitionVectorTests, TestStablePartitionStencilZipIterator)
{
    using Vector = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector data(5);
    data[0] = 1;
    data[1] = 0;
    data[2] = 1;
    data[3] = 1;
    data[4] = 0;

    Vector stencil1(5);
    Vector stencil2(5);

    stencil1[0] = 1;
    stencil2[0] = 2;
    stencil1[1] = 2;
    stencil2[1] = 0;
    stencil1[2] = 1;
    stencil2[2] = 3;
    stencil1[3] = 1;
    stencil2[3] = 2;
    stencil1[4] = 2;
    stencil2[4] = 1;

    using Iterator      = typename Vector::iterator;
    using IteratorTuple = typename thrust::tuple<Iterator, Iterator>;
    using ZipIterator   = typename thrust::zip_iterator<IteratorTuple>;

    ZipIterator stencil_begin
        = thrust::make_zip_iterator(thrust::make_tuple(stencil1.begin(), stencil2.begin()));

    Iterator mid = thrust::stable_partition(data.begin(), data.end(), stencil_begin, is_ordered());

    Vector ref(5);

    ref[0] = 1;
    ref[1] = 1;
    ref[2] = 1;
    ref[3] = 0;
    ref[4] = 0;

    ASSERT_EQ(ref, data);
    ASSERT_EQ(mid - data.begin(), 3);
}

template <typename ForwardIterator, typename Predicate>
__host__ __device__ ForwardIterator
                    partition(my_system& system, ForwardIterator first, ForwardIterator, Predicate)
{
    system.validate_dispatch();
    return first;
}

TEST(PartitionTests, TestPartitionDispatchExplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::partition(sys, vec.begin(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename InputIterator, typename Predicate>
__host__ __device__ ForwardIterator
                    partition(my_system& system, ForwardIterator first, ForwardIterator, InputIterator, Predicate)
{
    system.validate_dispatch();
    return first;
}

TEST(PartitionTests, TestPartitionStencilDispatchExplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::partition(sys, vec.begin(), vec.begin(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename Predicate>
__host__ __device__ ForwardIterator
                    partition(my_tag, ForwardIterator first, ForwardIterator, Predicate)
{
    *first = 13;
    return first;
}

TYPED_TEST(PartitionTests, TestPartitionDispatchImplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    thrust::partition(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0);

    ASSERT_EQ(13, vec.front());
}

template <typename ForwardIterator, typename InputIterator, typename Predicate>
__host__ __device__ ForwardIterator
                    partition(my_tag, ForwardIterator first, ForwardIterator, InputIterator, Predicate)
{
    *first = 13;
    return first;
}

TYPED_TEST(PartitionTests, TestPartitionStencilDispatchImplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    thrust::partition(thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      thrust::retag<my_tag>(vec.begin()),
                      0);

    ASSERT_EQ(13, vec.front());
}

template <typename InputIterator,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
__host__ __device__ thrust::pair<OutputIterator1, OutputIterator2>
                    partition_copy(my_system& system,
                                   InputIterator,
                                   InputIterator,
                                   OutputIterator1 out_true,
                                   OutputIterator2 out_false,
                                   Predicate)
{
    system.validate_dispatch();
    return thrust::make_pair(out_true, out_false);
}

TEST(PartitionTests, TestPartitionCopyDispatchExplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::partition_copy(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
__host__ __device__ thrust::pair<OutputIterator1, OutputIterator2>
                    partition_copy(my_system& system,
                                   InputIterator1,
                                   InputIterator1,
                                   InputIterator2,
                                   OutputIterator1 out_true,
                                   OutputIterator2 out_false,
                                   Predicate)
{
    system.validate_dispatch();
    return thrust::make_pair(out_true, out_false);
}

TEST(PartitionTests, TestPartitionCopyStencilDispatchExplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::partition_copy(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
__host__ __device__ thrust::pair<OutputIterator1, OutputIterator2>
                    partition_copy(my_tag,
                                   InputIterator first,
                                   InputIterator,
                                   OutputIterator1 out_true,
                                   OutputIterator2 out_false,
                                   Predicate)
{
    *first = 13;
    return thrust::make_pair(out_true, out_false);
}

TEST(PartitionTests, TestPartitionCopyDispatchImplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    thrust::partition_copy(thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           0);

    ASSERT_EQ(13, vec.front());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
__host__ __device__ thrust::pair<OutputIterator1, OutputIterator2>
                    partition_copy(my_tag,
                                   InputIterator1 first,
                                   InputIterator1,
                                   InputIterator2,
                                   OutputIterator1 out_true,
                                   OutputIterator2 out_false,
                                   Predicate)
{
    *first = 13;
    return thrust::make_pair(out_true, out_false);
}

TEST(PartitionTests, TestPartitionCopyStencilDispatchImplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    thrust::partition_copy(thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           thrust::retag<my_tag>(vec.begin()),
                           0);

    ASSERT_EQ(13, vec.front());
}

template <typename ForwardIterator, typename Predicate>
__host__ __device__ ForwardIterator
                    stable_partition(my_system& system, ForwardIterator first, ForwardIterator, Predicate)
{
    system.validate_dispatch();
    return first;
}

TEST(PartitionTests, TestStablePartitionDispatchExplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::stable_partition(sys, vec.begin(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename InputIterator, typename Predicate>
__host__ __device__ ForwardIterator stable_partition(
    my_system& system, ForwardIterator first, ForwardIterator, InputIterator, Predicate)
{
    system.validate_dispatch();
    return first;
}

TEST(PartitionTests, TestStablePartitionStencilDispatchExplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::stable_partition(sys, vec.begin(), vec.begin(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename Predicate>
__host__ __device__ ForwardIterator
                    stable_partition(my_tag, ForwardIterator first, ForwardIterator, Predicate)
{
    *first = 13;
    return first;
}

TEST(PartitionTests, TestStablePartitionDispatchImplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    thrust::stable_partition(
        thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0);

    ASSERT_EQ(13, vec.front());
}

template <typename ForwardIterator, typename InputIterator, typename Predicate>
__host__ __device__ ForwardIterator
                    stable_partition(my_tag, ForwardIterator first, ForwardIterator, InputIterator, Predicate)
{
    *first = 13;
    return first;
}

TEST(PartitionTests, TestStablePartitionStencilDispatchImplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    thrust::stable_partition(thrust::retag<my_tag>(vec.begin()),
                             thrust::retag<my_tag>(vec.begin()),
                             thrust::retag<my_tag>(vec.begin()),
                             0);

    ASSERT_EQ(13, vec.front());
}

template <typename InputIterator,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
__host__ __device__ thrust::pair<OutputIterator1, OutputIterator2>
                    stable_partition_copy(my_system& system,
                                          InputIterator,
                                          InputIterator,
                                          OutputIterator1 out_true,
                                          OutputIterator2 out_false,
                                          Predicate)
{
    system.validate_dispatch();
    return thrust::make_pair(out_true, out_false);
}

TEST(PartitionTests, TestStablePartitionCopyDispatchExplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::stable_partition_copy(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
__host__ __device__ thrust::pair<OutputIterator1, OutputIterator2>
                    stable_partition_copy(my_system& system,
                                          InputIterator1,
                                          InputIterator1,
                                          InputIterator2,
                                          OutputIterator1 out_true,
                                          OutputIterator2 out_false,
                                          Predicate)
{
    system.validate_dispatch();
    return thrust::make_pair(out_true, out_false);
}

TEST(PartitionTests, TestStablePartitionCopyStencilDispatchExplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::stable_partition_copy(
        sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
__host__ __device__ thrust::pair<OutputIterator1, OutputIterator2>
                    stable_partition_copy(my_tag,
                                          InputIterator first,
                                          InputIterator,
                                          OutputIterator1 out_true,
                                          OutputIterator2 out_false,
                                          Predicate)
{
    *first = 13;
    return thrust::make_pair(out_true, out_false);
}

TEST(PartitionTests, TestStablePartitionCopyDispatchImplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    thrust::stable_partition_copy(thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()),
                                  0);

    ASSERT_EQ(13, vec.front());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename Predicate>
__host__ __device__ thrust::pair<OutputIterator1, OutputIterator2>
                    stable_partition_copy(my_tag,
                                          InputIterator1 first,
                                          InputIterator1,
                                          InputIterator2,
                                          OutputIterator1 out_true,
                                          OutputIterator2 out_false,
                                          Predicate)
{
    *first = 13;
    return thrust::make_pair(out_true, out_false);
}

TEST(PartitionTests, TestStablePartitionCopyStencilDispatchImplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    thrust::stable_partition_copy(thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()),
                                  thrust::retag<my_tag>(vec.begin()),
                                  0);

    ASSERT_EQ(13, vec.front());
}

__global__
THRUST_HIP_LAUNCH_BOUNDS_DEFAULT
void PartitionKernel(int const N, int* array)
{
    if(threadIdx.x == 0)
    {
        thrust::device_ptr<int> begin(array);
        thrust::device_ptr<int> end(array + N);
        thrust::partition(thrust::hip::par, begin, end,is_even<int>());

    }
}

TEST(PartitionTests,TestPartitionDevice)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

      for(auto size : get_sizes() )
      {
          SCOPED_TRACE(testing::Message() << "with size= " << size);

          for(auto seed : get_seeds())
          {
              SCOPED_TRACE(testing::Message() << "with seed= " << seed);

              thrust::host_vector<int> h_data = get_random_data<int>(size, 0, size, seed);

              thrust::device_vector<int> d_data = h_data;

              thrust::partition(h_data.begin(), h_data.end(),is_even<int>());

              hipLaunchKernelGGL(PartitionKernel,
                                 dim3(1, 1, 1),
                                 dim3(128, 1, 1),
                                 0,
                                 0,
                                 size,
                                 thrust::raw_pointer_cast(&d_data[0]));

              ASSERT_EQ(h_data, d_data);
          }
      }
}

__global__
THRUST_HIP_LAUNCH_BOUNDS_DEFAULT
void PartitionCopyKernel(int const N, int* array, int* true_arr, int* false_arr, int* size_array)
{
    if(threadIdx.x == 0)
    {
        thrust::device_ptr<int> begin(array);
        thrust::device_ptr<int> end(array + N);
        thrust::device_ptr<int> true_begin(true_arr);
        thrust::device_ptr<int> false_begin(false_arr);
        typedef thrust::device_vector<int>::iterator iter;

        thrust::pair<iter,iter> end_pair = thrust::partition_copy(thrust::hip::par, begin, end, true_begin, false_begin, is_even<int>());
        size_array[0] = end_pair.first - iter(true_begin);
        size_array[1] = end_pair.second - iter(false_begin);


    }
}

TEST(PartitionTests,TestPartitionCopyDevice)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

      for(auto size : get_sizes() )
      {
          SCOPED_TRACE(testing::Message() << "with size= " << size);

          for(auto seed : get_seeds())
          {
              SCOPED_TRACE(testing::Message() << "with seed= " << seed);

              thrust::host_vector<int> h_data = get_random_data<int>(size, 0, size, seed);
              thrust::host_vector<int> h_true(size);
              thrust::host_vector<int> h_false(size);

              thrust::device_vector<int> d_data = h_data;
              thrust::device_vector<int> d_true = h_true;
              thrust::device_vector<int> d_false = h_false;
              thrust::device_vector<int> d_output_sizes(2,0);

              auto end_pair =  thrust::partition_copy(h_data.begin(), h_data.end(), h_true.begin(), h_false.begin(), is_even<int>());
              h_true.resize(end_pair.first - h_true.begin());
              h_false.resize(end_pair.second - h_false.begin());

              hipLaunchKernelGGL(PartitionCopyKernel,
                                 dim3(1, 1, 1),
                                 dim3(128, 1, 1),
                                 0,
                                 0,
                                 size,
                                 thrust::raw_pointer_cast(&d_data[0]),
                                 thrust::raw_pointer_cast(&d_true[0]),
                                 thrust::raw_pointer_cast(&d_false[0]),
                                 thrust::raw_pointer_cast(&d_output_sizes[0]));

              d_true.resize(d_output_sizes[0]);
              d_false.resize(d_output_sizes[1]);

              ASSERT_EQ(h_true, d_true);
              ASSERT_EQ(h_false, d_false);
          }
      }
}

//TODO: Tests fails with error message "Memory access fault by GPU node-1 (Agent handle: 0x1ade7d0) on address 0x7fac25a00000. Reason: Page not present or supervisor privilege."

// __global__
// THRUST_HIP_LAUNCH_BOUNDS_DEFAULT
// void StablePartitionKernel(int const N, int* array)
// {
//     if(threadIdx.x == 0)
//     {
//         thrust::device_ptr<int> begin(array);
//         thrust::device_ptr<int> end(array + N);
//         //TODO: The thrust::hip::par throw exception, we should fix it
//         thrust::stable_partition(thrust::hip::par, begin, end,is_even<int>());
//
//     }
// }
//
// TEST(PartitionTests,TestStablePartitionDevice)
// {
//     SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
//
//       for(auto size : get_sizes() )
//       {
//           SCOPED_TRACE(testing::Message() << "with size= " << size);
//
//           for(auto seed : get_seeds())
//           {
//               SCOPED_TRACE(testing::Message() << "with seed= " << seed);
//
//               thrust::host_vector<int> h_data = get_random_data<int>(size, 0, size, seed);
//               thrust::device_vector<int> d_data = h_data;
//
//               thrust::stable_partition(h_data.begin(), h_data.end(),is_even<int>());
//
//               hipLaunchKernelGGL(StablePartitionKernel,
//                                  dim3(1, 1, 1),
//                                  dim3(128, 1, 1),
//                                  0,
//                                  0,
//                                  size,
//                                  thrust::raw_pointer_cast(&d_data[0]));
//
//               ASSERT_EQ(h_data, d_data);
//           }
//       }
// }
