#include <ttl/tensor.hpp>
#include <cassert>
#include <mdspan>

#undef DNDEBUG

using namespace ttl::concepts;

static_assert([] {
    static_assert(scalar<int>);
    static_assert(scalar<int const>);
    static_assert(ttl::evaluate(42) == 42);

    static_assert(scalar<float>);
    static_assert(scalar<float const>);
    static_assert(ttl::evaluate(3.1415) == 3.1415);

    static_assert(ttl::evaluate(3.1415, std::array<std::size_t, 0>{}) == 3.1415);

    static_assert(ttl::extents(0) == std::extents<std::size_t>{});
    return true;
 }());

static_assert([] {
    static_assert(tensor_of_rank<const int[3], 1>);
    constexpr int x[] = {1, 2, 3};
    static_assert(ttl::evaluate(x, 0) == 1);
    static_assert(ttl::evaluate(x, 1) == 2);
    static_assert(ttl::evaluate(x, 2) == 3);

    static_assert(tensor_of_rank<int[3], 1>);
    int y[] = {1, 2, 3};
    assert(ttl::evaluate(y, 0) == 1);
    assert(ttl::evaluate(y, 1) == 2);
    assert(ttl::evaluate(y, 2) == 3);

    ttl::evaluate(y, 0) = 4;
    ttl::evaluate(y, 1) = 5;
    ttl::evaluate(y, 2) = 6;

    assert(ttl::evaluate(y, 0) == 4);
    assert(ttl::evaluate(y, 1) == 5);
    assert(ttl::evaluate(y, 2) == 6);
    return true;
 }());

static_assert([] {
    static_assert(tensor_of_rank<const int[3][1], 2>);
    constexpr int x[][1] = {
        {1}, {2}, {3}
    };
    static_assert(ttl::evaluate(x, 0, 0) == 1);
    static_assert(ttl::evaluate(x, 1, 0) == 2);
    static_assert(ttl::evaluate(x, 2, 0) == 3);

    static_assert(tensor_of_rank<int[3][1], 2>);
    int y[][1] = {
        {1}, {2}, {3}
    };
    assert(ttl::evaluate(y, 0, 0) == 1);
    assert(ttl::evaluate(y, 1, 0) == 2);
    assert(ttl::evaluate(y, 2, 0) == 3);

    ttl::evaluate(y, 0, 0) = 4;
    ttl::evaluate(y, 1, 0) = 5;
    ttl::evaluate(y, 2, 0) = 6;

    assert(ttl::evaluate(y, 0, 0) == 4);
    assert(ttl::evaluate(y, 1, 0) == 5);
    assert(ttl::evaluate(y, 2, 0) == 6);
    return true;
 }());

static_assert([] {
    static_assert(tensor_of_rank<const std::array<int, 3>, 1>);
    constexpr std::array x{1, 2, 3};
    static_assert(ttl::evaluate(x, 0) == 1);
    static_assert(ttl::evaluate(x, 1) == 2);
    static_assert(ttl::evaluate(x, 2) == 3);

    static_assert(tensor_of_rank<std::array<int, 3>, 1>);
    std::array y{1, 2, 3};
    assert(ttl::evaluate(y, 0) == 1);
    assert(ttl::evaluate(y, 1) == 2);
    assert(ttl::evaluate(y, 2) == 3);
    return true;
 }());


static_assert([] {
    static_assert(tensor_of_rank<const std::array<std::array<int, 1>, 3>, 2>);
    constexpr std::array x {
        std::array{1},
        std::array{2},
        std::array{3}
    };
    static_assert(ttl::evaluate(x, 0, 0) == 1);
    static_assert(ttl::evaluate(x, 1, 0) == 2);
    static_assert(ttl::evaluate(x, 2, 0) == 3);

    static_assert(tensor_of_rank<std::array<std::array<int, 1>, 3>, 2>);
    std::array y {
        std::array{1},
        std::array{2},
        std::array{3}
    };
    assert(ttl::evaluate(y, 0, 0) == 1);
    assert(ttl::evaluate(y, 1, 0) == 2);
    assert(ttl::evaluate(y, 2, 0) == 3);
    return true;
 }());

static_assert([] {
    static_assert(tensor_of_rank<std::vector<int>, 1>);
    std::vector x{1, 2, 3};
    assert(ttl::evaluate(x, 0) == 1);
    assert(ttl::evaluate(x, 1) == 2);
    assert(ttl::evaluate(x, 2) == 3);

    static_assert(tensor_of_rank<const std::vector<int>, 1>);
    const std::vector y{1, 2, 3};
    assert(ttl::evaluate(y, 0) == 1);
    assert(ttl::evaluate(y, 1) == 2);
    assert(ttl::evaluate(y, 2) == 3);
    return true;
 }());

static_assert([] {
    static_assert(tensor_of_rank<std::vector<std::array<int, 1>>, 2>);
    std::vector x {
        std::array{1},
        std::array{2},
        std::array{3}
    };
    assert(ttl::evaluate(x, 0, 0) == 1);
    assert(ttl::evaluate(x, 1, 0) == 2);
    assert(ttl::evaluate(x, 2, 0) == 3);

    static_assert(tensor_of_rank<const std::vector<std::array<int, 1>>, 2>);
    const std::vector y {
        std::array{1},
        std::array{2},
        std::array{3}
    };
    assert(ttl::evaluate(y, 0, 0) == 1);
    assert(ttl::evaluate(y, 1, 0) == 2);
    assert(ttl::evaluate(y, 2, 0) == 3);

    return true;
 }());

static_assert([] {
    static constexpr int x[]{1, 2, 3};
    constexpr auto s = std::mdspan(x, std::extents<int, 3>());
    static_assert(tensor_of_rank<decltype(auto(s)), 1>);
    static_assert(tensor_of_rank<decltype(s), 1>);
    static_assert(ttl::evaluate(s, 0) == 1);
    static_assert(ttl::evaluate(s, 1) == 2);
    static_assert(ttl::evaluate(s, 2) == 3);

    int y[]{1, 2, 3};
    auto t = std::mdspan(y, 3);
    static_assert(tensor_of_rank<decltype(auto(t)), 1>);
    static_assert(tensor_of_rank<decltype(t), 1>);

    assert(ttl::evaluate(t, 0) == 1);
    assert(ttl::evaluate(t, 1) == 2);
    assert(ttl::evaluate(t, 2) == 3);

    ttl::evaluate(t, 0) = 4;
    ttl::evaluate(t, 1) = 5;
    ttl::evaluate(t, 2) = 6;

    assert(ttl::evaluate(t, 0) == 4);
    assert(ttl::evaluate(t, 1) == 5);
    assert(ttl::evaluate(t, 2) == 6);
    return true;
 }());

static_assert([] {
    static constexpr int x[]{1, 2, 3};
    constexpr auto s = std::mdspan(x, std::extents<int, 3, 1>());
    static_assert(tensor_of_rank<decltype(auto(s)), 2>);
    static_assert(tensor_of_rank<decltype(s), 2>);
    static_assert(ttl::evaluate(s, 0, 0) == 1);
    static_assert(ttl::evaluate(s, 1, 0) == 2);
    static_assert(ttl::evaluate(s, 2, 0) == 3);

    int y[]{1, 2, 3};
    auto t = std::mdspan(y, 3, 1);
    static_assert(tensor_of_rank<decltype(auto(t)), 2>);
    static_assert(tensor_of_rank<decltype(t), 2>);

    assert(ttl::evaluate(t, 0, 0) == 1);
    assert(ttl::evaluate(t, 1, 0) == 2);
    assert(ttl::evaluate(t, 2, 0) == 3);

    ttl::evaluate(t, 0, 0) = 4;
    ttl::evaluate(t, 1, 0) = 5;
    ttl::evaluate(t, 2, 0) = 6;

    assert(ttl::evaluate(t, 0, 0) == 4);
    assert(ttl::evaluate(t, 1, 0) == 5);
    assert(ttl::evaluate(t, 2, 0) == 6);
    return true;
 }());
