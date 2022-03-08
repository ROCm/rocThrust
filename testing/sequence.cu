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
 
#include <unittest/unittest.h>
#include <thrust/sequence.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>


template<typename ForwardIterator>
__host__ __device__
void sequence(my_system &system, ForwardIterator, ForwardIterator)
{
    system.validate_dispatch();
}

void TestSequenceDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::sequence(sys, vec.begin(), vec.end());

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestSequenceDispatchExplicit);


template<typename ForwardIterator>
__host__ __device__
void sequence(my_tag, ForwardIterator first, ForwardIterator)
{
    *first = 13;
}

void TestSequenceDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::sequence(thrust::retag<my_tag>(vec.begin()),
                     thrust::retag<my_tag>(vec.end()));

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestSequenceDispatchImplicit);


template <class Vector>
void TestSequenceSimple()
{
    using value_type = typename Vector::value_type;
    Vector v(5);

    thrust::sequence(v.begin(), v.end());

    ASSERT_EQUAL(v[0], 0);
    ASSERT_EQUAL(v[1], 1);
    ASSERT_EQUAL(v[2], 2);
    ASSERT_EQUAL(v[3], 3);
    ASSERT_EQUAL(v[4], 4);

    thrust::sequence(v.begin(), v.end(), value_type{10});

    ASSERT_EQUAL(v[0], 10);
    ASSERT_EQUAL(v[1], 11);
    ASSERT_EQUAL(v[2], 12);
    ASSERT_EQUAL(v[3], 13);
    ASSERT_EQUAL(v[4], 14);
    
    thrust::sequence(v.begin(), v.end(), value_type{10}, value_type{2});

    ASSERT_EQUAL(v[0], 10);
    ASSERT_EQUAL(v[1], 12);
    ASSERT_EQUAL(v[2], 14);
    ASSERT_EQUAL(v[3], 16);
    ASSERT_EQUAL(v[4], 18);
}
DECLARE_VECTOR_UNITTEST(TestSequenceSimple);


template <typename T>
void TestSequence(size_t n)
{
    thrust::host_vector<T>   h_data(n);
    thrust::device_vector<T> d_data(n);

    thrust::sequence(h_data.begin(), h_data.end());
    thrust::sequence(d_data.begin(), d_data.end());

    ASSERT_EQUAL(h_data, d_data);

    thrust::sequence(h_data.begin(), h_data.end(), T(10));
    thrust::sequence(d_data.begin(), d_data.end(), T(10));

    ASSERT_EQUAL(h_data, d_data);

    thrust::sequence(h_data.begin(), h_data.end(), T(10), T(2));
    thrust::sequence(d_data.begin(), d_data.end(), T(10), T(2));

    ASSERT_EQUAL(h_data, d_data);

    thrust::sequence(h_data.begin(), h_data.end(), T(10), T(2));
    thrust::sequence(d_data.begin(), d_data.end(), T(10), T(2));

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestSequence);


template <typename T>
void TestSequenceToDiscardIterator(size_t n)
{
    thrust::host_vector<T>   h_data(n);
    thrust::device_vector<T> d_data(n);

    thrust::sequence(thrust::discard_iterator<thrust::device_system_tag>(),
                     thrust::discard_iterator<thrust::device_system_tag>(13),
                     T(10),
                     T(2));

    // nothing to check -- just make sure it compiles
}
DECLARE_VARIABLE_UNITTEST(TestSequenceToDiscardIterator);


void TestSequenceComplex()
{
  thrust::device_vector<thrust::complex<double> > m(64);
  thrust::sequence(m.begin(), m.end());
}
DECLARE_UNITTEST(TestSequenceComplex);

// A class that doesnt accept conversion from size_t but can be multiplied by a scalar
struct Vector
{
    Vector() = default;
    // Explicitly disable construction from size_t
    Vector(std::size_t) = delete;
    __host__ __device__ Vector(int x_, int y_) : x{x_}, y{y_} {}
    Vector(const Vector&) = default;
    Vector &operator=(const Vector&) = default;

    int x, y;
};

// Vector-Vector addition
__host__ __device__ Vector operator+(const Vector a, const Vector b) { return Vector{a.x + b.x, a.y + b.y}; }
// Vector-Scalar Multiplication
__host__ __device__ Vector operator*(const int a, const Vector b) { return Vector{a * b.x, a * b.y}; }
__host__ __device__ Vector operator*(const Vector b, const int a) { return Vector{a * b.x, a * b.y}; }

void TestSequenceNoSizeTConversion()
{
    thrust::device_vector<Vector> m(64);
    thrust::sequence(m.begin(), m.end(), ::Vector{0, 0}, ::Vector{1, 2});

    for (std::size_t i = 0; i < m.size(); ++i)
    {
        const ::Vector v = m[i];
        ASSERT_EQUAL(static_cast<std::size_t>(v.x), i);
        ASSERT_EQUAL(static_cast<std::size_t>(v.y), 2 * i);
    }
}
DECLARE_UNITTEST(TestSequenceNoSizeTConversion);
