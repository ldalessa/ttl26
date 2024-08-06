#include <ttl/std.hpp>
#include <ttl/tensor.hpp>
#include <ttl/tensor_index.hpp>
#include <ttl/tree/bind.hpp>
#include <cassert>

#undef DNDEBUG

static constexpr ttl::tensor_index i = "i";
static constexpr ttl::tensor_index j = "j";
static constexpr ttl::tensor_index ii = i + i;
static constexpr ttl::tensor_index ij = i + j;

static constexpr bool test_vector()
{
    int x[3] = {
        0, 1, 2
    };

    auto bx = ttl::bind<i>(x);
    static_assert(ttl::rank<decltype(bx)> == 1);
    static_assert(decltype(bx)::_inner.rank() == 1);
    assert(bx[0] == 0);
    assert(bx[1] == 1);
    assert(bx[2] == 2);

    bx[0] += 1;
    bx[1] += 1;
    bx[2] += 1;

    assert(bx[0] == 1);
    assert(bx[1] == 2);
    assert(bx[2] == 3);

    return true;
}

static_assert(test_vector());

static constexpr bool test_matrix()
{
    int A[3][3] = {
        {0, 1, 2},
        {3, 4, 5},
        {6, 7, 8}
    };

    auto bx = ttl::bind<ij>(A);
    static_assert(ttl::rank<decltype(bx)> == 2);
    static_assert(decltype(bx)::_inner.rank() == 2);
    assert((bx[0,0] == 0));
    assert((bx[1,0] == 3));
    assert((bx[2,2] == 8));

    auto cx = ttl::bind<ii>(A);
    static_assert(ttl::rank<decltype(cx)> == 0);
    assert(cx[] == 12);
    assert(cx == 12);

    int const B[6] = {
        0, 1,
        2, 3,
        4, 5
    };
    auto bspan = std::mdspan(B, 3, 2);
    auto bb = ttl::bind<ij>(bspan);
    assert((bb[0,0] == 0));
    assert((bb[0,1] == 1));
    assert((bb[1,0] == 2));
    assert((bb[1,1] == 3));
    assert((bb[2,0] == 4));
    assert((bb[2,1] == 5));

    return true;
}

static_assert(test_matrix());
