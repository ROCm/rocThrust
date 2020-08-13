// Copyright (c) 2011, David Thomas
//
// Copyright(c) 2018 M�t� Ferenc Nagy-Egri, Wigner GPU-Laboratory.
//
// All rights reserved.
//
// The 3-clause BSD License is applied to this software, see LICENSE.txt
//

#pragma once

#ifdef __SYCL_DEVICE_ONLY__
// SYCL include
#include <CL/sycl.hpp>
#endif

// Standard C++ includes
#include <cstddef>  // std::size_t
#include <cstdint>  // std::uint64_t
#include <limits>   // std::numeric_limits::min,max
#include <array>    // std::array

namespace prng
{
    template <std::uint32_t A, std::uint64_t M>
    class multiply_with_carry_engine_32
    {
    public:

        using result_type = std::uint32_t;

        static constexpr std::size_t word_size = 32;
        static constexpr std::size_t state_size = 2;
        static constexpr result_type mask = 0xffffffff;

        static constexpr result_type default_seed = 5489u;

        multiply_with_carry_engine_32(result_type value) { seed(value); }

        //template <typename Sseq> explicit multiply_with_carry_engine_32(Sseq& s);

        multiply_with_carry_engine_32() : multiply_with_carry_engine_32(default_seed) {}
        multiply_with_carry_engine_32(const multiply_with_carry_engine_32&) = default;

        void seed(result_type value = default_seed)
        {
        }
        //template <typename Sseq> void seed(Sseq& s);

        result_type operator()()
        {
            next_state();

            return x ^ c;
        }

        void discard(unsigned long long z)
        {
            auto tmp = skip_impl_mod64({ x, c }, z);
            x = tmp[0];
            c = tmp[1];
        }

        friend bool operator==(const multiply_with_carry_engine_32<A, M>& lhs,
                               const multiply_with_carry_engine_32<A, M>& rhs)
        {
            return (lhs.x == rhs.x) &&
                   (lhs.c == rhs.c);
        }

        friend bool operator!=(const multiply_with_carry_engine_32<A, M>& lhs,
                               const multiply_with_carry_engine_32<A, M>& rhs)
        {
            return (lhs.x != rhs.x) ||
                   (lhs.c != rhs.c);
        }

        static constexpr result_type min() { return std::numeric_limits<result_type>::min(); }
        static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    private:

        result_type x, c;

        // Convenience renames
        static constexpr auto a = A;
        static constexpr auto m = M;

        inline void next_state()
        {
#ifdef __SYCL_DEVICE_ONLY__
            std::uint32_t xn = a * x + c;
            std::uint32_t carry = static_cast<std::uint32_t>(xn < c); // The (Xn<C) will be zero or one for scalar
            std::uint32_t cn = cl::sycl::mad_hi(a, x, carry);

            x = xn;
            c = cn;
#else
            *reinterpret_cast<std::uint64_t*>(this) = x * static_cast<std::uint64_t>(a) + c;
#endif
        }

        std::uint64_t add_mod64(std::uint64_t a_,
                                std::uint64_t b_,
                                std::uint64_t M_)
        {
            std::uint64_t v_ = a_ + b_;
            if ((v_ >= M_) || (v_ < a_))
                v_ = v_ - M_;
            return v_;
        }

        std::uint64_t mul_mod64(std::uint64_t a_,
                                std::uint64_t b_,
                                std::uint64_t M_)
        {
            std::uint64_t r_ = 0;
            while (a_ != 0) {
                if (a_ & 1)
                    r_ = add_mod64(r_, b_, M_);
                b_ = add_mod64(b_, b_, M_);
                a_ = a_ >> 1;
            }
            return r_;
        }

        std::uint64_t pow_mod64(std::uint64_t a_,
                                std::uint64_t e_,
                                std::uint64_t M_)
        {
            std::uint64_t sqr_ = a_, acc_ = 1;
            while (e_ != 0) {
                if (e_ & 1)
                    acc_ = mul_mod64(acc_, sqr_, M_);
                sqr_ = mul_mod64(sqr_, sqr_, M_);
                e_ = e_ >> 1;
            }
            return acc_;
        }

        std::array<std::uint32_t, 2> skip_impl_mod64(std::array<std::uint32_t, 2> curr_,
                                                     std::uint64_t distance_)
        {
            std::uint64_t m_ = pow_mod64(a, distance_, m);
            std::uint64_t x = curr_[0]*(std::uint64_t)a + curr_[1];
            x = mul_mod64(x, m_, m);
            return { (std::uint32_t)(x / a),
                     (std::uint32_t)(x % a) };
        }
    };

    using mwc64x_32 = multiply_with_carry_engine_32<4294883355u, 18446383549859758079ul>;
}