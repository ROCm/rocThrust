#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "test_header.hpp"

template<typename BaseAlloc, bool PropagateOnSwap>
class stateful_allocator : public BaseAlloc
{
public:
    stateful_allocator(int i) : state(i)
    {
    }

    ~stateful_allocator() {}

    stateful_allocator(const stateful_allocator &other)
        : BaseAlloc(other), state(other.state)
    {
    }

    stateful_allocator & operator=(const stateful_allocator & other)
    {
        state = other.state;
        return *this;
    }

    stateful_allocator(stateful_allocator && other)
        : BaseAlloc(std::move(other)), state(other.state)
    {
        other.state = 0;
    }

    stateful_allocator & operator=(stateful_allocator && other)
    {
        state = other.state;
        other.state = 0;
        return *this;
    }

    static int last_allocated;
    static int last_deallocated;

    typedef
        typename thrust::detail::allocator_traits<BaseAlloc>::pointer
        pointer;

    pointer allocate(std::size_t size)
    {
        last_allocated = state;
        return BaseAlloc::allocate(size);
    }

    void deallocate(pointer ptr, std::size_t size)
    {
        last_deallocated = state;
        return BaseAlloc::deallocate(ptr, size);
    }

    bool operator==(const stateful_allocator &rhs) const
    {
        return state == rhs.state;
    }

    bool operator!=(const stateful_allocator &rhs) const
    {
        return state != rhs.state;
    }

    friend std::ostream & operator<<(std::ostream &os,
        const stateful_allocator & alloc)
    {
        os << "stateful_alloc(" << alloc.state << ")";
        return os;
    }

    typedef thrust::detail::false_type is_always_equal;
    typedef thrust::detail::true_type propagate_on_container_copy_assignment;
    typedef thrust::detail::true_type propagate_on_container_move_assignment;
    typedef thrust::detail::integral_constant<bool, PropagateOnSwap> propagate_on_container_swap;

private:
    int state;
};

template<typename BaseAlloc, bool PropagateOnSwap>
int stateful_allocator<BaseAlloc, PropagateOnSwap>::last_allocated = 0;

template<typename BaseAlloc, bool PropagateOnSwap>
int stateful_allocator<BaseAlloc, PropagateOnSwap>::last_deallocated = 0;

typedef stateful_allocator<std::allocator<int>, true> host_alloc;
typedef stateful_allocator<thrust::device_allocator<int>, true> device_alloc;

typedef thrust::host_vector<int, host_alloc> host_vector;
typedef thrust::device_vector<int, device_alloc> device_vector;

typedef stateful_allocator<std::allocator<int>, false> host_alloc_nsp;
typedef stateful_allocator<thrust::device_allocator<int>, false> device_alloc_nsp;

typedef thrust::host_vector<int, host_alloc_nsp> host_vector_nsp;
typedef thrust::device_vector<int, device_alloc_nsp> device_vector_nsp;

template<typename Vector>
void TestVectorAllocatorConstructors()
{
    typedef typename Vector::allocator_type Alloc;
    Alloc alloc1(1);
    Alloc alloc2(2);

    Vector v1(alloc1);
    ASSERT_EQ(v1.get_allocator(), alloc1);

    Vector v2(10, alloc1);
    ASSERT_EQ(v2.size(), 10u);
    ASSERT_EQ(v2.get_allocator(), alloc1);
    ASSERT_EQ(Alloc::last_allocated, 1);
    Alloc::last_allocated = 0;

    Vector v3(10, 17, alloc1);
    ASSERT_EQ((v3 == std::vector<int>(10, 17)), true);
    ASSERT_EQ(v3.get_allocator(), alloc1);
    ASSERT_EQ(Alloc::last_allocated, 1);
    Alloc::last_allocated = 0;

    Vector v4(v3, alloc2);
    ASSERT_EQ((v3 == v4), true);
    ASSERT_EQ(v4.get_allocator(), alloc2);
    ASSERT_EQ(Alloc::last_allocated, 2);
    Alloc::last_allocated = 0;

    // FIXME: uncomment this after the vector_base(vector_base&&, const Alloc&)
    // is fixed and implemented
    // Vector v5(std::move(v3), alloc2);
    // ASSERT_EQ((v4 == v5), true);
    // ASSERT_EQ(v5.get_allocator(), alloc2);
    // ASSERT_EQ(Alloc::last_allocated, 1);
    // Alloc::last_allocated = 0;

    Vector v6(v4.begin(), v4.end(), alloc2);
    ASSERT_EQ((v4 == v6), true);
    ASSERT_EQ(v6.get_allocator(), alloc2);
    ASSERT_EQ(Alloc::last_allocated, 2);
}

TEST(VectorAllocatorTests, TestVectorAllocatorConstructorsHost)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestVectorAllocatorConstructors<host_vector>();
}

TEST(VectorAllocatorTests, TestVectorAllocatorConstructorsDevice)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestVectorAllocatorConstructors<device_vector>();
}

template<typename Vector>
void TestVectorAllocatorPropagateOnCopyAssignment()
{
    ASSERT_EQ(thrust::detail::allocator_traits<typename Vector::allocator_type>::propagate_on_container_copy_assignment::value, true);

    typedef typename Vector::allocator_type Alloc;
    Alloc alloc1(1);
    Alloc alloc2(2);

    Vector v1(10, alloc1);
    Vector v2(15, alloc2);

    v2 = v1;
    ASSERT_EQ((v1 == v2), true);
    ASSERT_EQ(v2.get_allocator(), alloc1);
    ASSERT_EQ(Alloc::last_allocated, 1);
    ASSERT_EQ(Alloc::last_deallocated, 2);
}

TEST(VectorAllocatorTests, TestVectorAllocatorPropagateOnCopyAssignmentHost)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestVectorAllocatorPropagateOnCopyAssignment<host_vector>();
}

TEST(VectorAllocatorTests, TestVectorAllocatorPropagateOnCopyAssignmentDevice)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestVectorAllocatorPropagateOnCopyAssignment<device_vector>();
}

template<typename Vector>
void TestVectorAllocatorPropagateOnMoveAssignment()
{
    typedef typename Vector::allocator_type Alloc;
    ASSERT_EQ(thrust::detail::allocator_traits<typename Vector::allocator_type>::propagate_on_container_copy_assignment::value, true);

    typedef typename Vector::allocator_type Alloc;
    Alloc alloc1(1);
    Alloc alloc2(2);

    {
    Vector v1(10, alloc1);
    Vector v2(15, alloc2);

    v2 = std::move(v1);
    ASSERT_EQ(v2.get_allocator(), alloc1);
    ASSERT_EQ(Alloc::last_allocated, 2);
    ASSERT_EQ(Alloc::last_deallocated, 2);
    }

    ASSERT_EQ(Alloc::last_deallocated, 1);
}

TEST(VectorAllocatorTests, TestVectorAllocatorPropagateOnMoveAssignmentHost)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestVectorAllocatorPropagateOnMoveAssignment<host_vector>();
}

TEST(VectorAllocatorTests, TestVectorAllocatorPropagateOnMoveAssignmentDevice)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestVectorAllocatorPropagateOnMoveAssignment<device_vector>();
}

template<typename Vector>
void TestVectorAllocatorPropagateOnSwap()
{
    typedef typename Vector::allocator_type Alloc;
    Alloc alloc1(1);
    Alloc alloc2(2);

    Vector v1(10, alloc1);
    Vector v2(17, alloc1);
    thrust::swap(v1, v2);

    ASSERT_EQ(v1.size(), 17u);
    ASSERT_EQ(v2.size(), 10u);

    Vector v3(15, alloc1);
    Vector v4(31, alloc2);
    ASSERT_THROW(thrust::swap(v3, v4), thrust::detail::allocator_mismatch_on_swap);
}

TEST(VectorAllocatorTests, TestVectorAllocatorPropagateOnSwapHost)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    
    TestVectorAllocatorPropagateOnSwap<host_vector_nsp>();
}

TEST(VectorAllocatorTests, TestVectorAllocatorPropagateOnSwapDevice)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    
    TestVectorAllocatorPropagateOnSwap<device_vector_nsp>();
}
