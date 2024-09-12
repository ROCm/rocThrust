#include <unittest/unittest.h>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

struct detect_wrong_difference
{
    bool * flag;

    THRUST_HOST_DEVICE detect_wrong_difference operator++() const { return *this; }
    THRUST_HOST_DEVICE detect_wrong_difference operator*() const { return *this; }
    template<typename Difference>
    THRUST_HOST_DEVICE detect_wrong_difference operator+(Difference) const { return *this; }
    template<typename Index>
    THRUST_HOST_DEVICE detect_wrong_difference operator[](Index) const { return *this; }

    THRUST_DEVICE
    void operator=(long long difference) const
    {
        if (difference != 1)
        {
            *flag = false;
        }
    }
};

void TestAdjacentDifferenceWithBigIndexesHelper(int magnitude)
{
    thrust::counting_iterator<long long> begin(1);
    thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
    ASSERT_EQUAL(thrust::distance(begin, end), 1ll << magnitude);

    thrust::device_ptr<bool> all_differences_correct = thrust::device_malloc<bool>(1);
    *all_differences_correct = true;

    detect_wrong_difference out = { thrust::raw_pointer_cast(all_differences_correct) };

    thrust::adjacent_difference(thrust::device, begin, end, out);

    bool all_differences_correct_h = *all_differences_correct;
    thrust::device_free(all_differences_correct);

    ASSERT_EQUAL(all_differences_correct_h, true);
}

void TestAdjacentDifferenceWithBigIndexes()
{
    TestAdjacentDifferenceWithBigIndexesHelper(30);
    TestAdjacentDifferenceWithBigIndexesHelper(31);
    TestAdjacentDifferenceWithBigIndexesHelper(32);
    TestAdjacentDifferenceWithBigIndexesHelper(33);
}
DECLARE_UNITTEST(TestAdjacentDifferenceWithBigIndexes);
