#undef DNDEBUG

#include <ttl/ttl.hpp>

using namespace ttl::literals;

static constexpr auto i = "i"_id;
static constexpr auto j = "j"_id;

static constexpr bool _scalars()
{
    int n = 0;
    auto const m = ttl::bind(n);

    m = 1;
    assert(n == 1);

    m = 2.;
    assert(n == 2);

    m = ttl::bind(3);
    assert(n == 3);

    int const A[4] { 1, 0, 0, 1 };
    auto const B = ttl::tspan(A, 2, 2);
    m = B(i, i);
    assert(n == 2);

    m += B(i, i);
    assert(n == 4);

    m -= B(i, i);
    assert(n == 2);

    m *= 42;
    assert(n == 84);

    return true;
}

static constexpr bool _vectors()
{
    int x[] { 0, 0 };
    int y[] { 1, 2 };
    ttl::bind(x, i) = ttl::bind(y, i);
    assert(x[0] == y[0]);
    assert(x[1] == y[1]);

    ttl::tspan(x, 2) << 2 * ttl::bind(y, i);
    assert(x[0] == 2 * y[0]);
    assert(x[1] == 2 * y[1]);

    std::mdspan(x, 2) << ttl::bind(y, i);
    assert(x[0] == y[0]);
    assert(x[1] == y[1]);

    int A[][2] { { 1, 2 }, { 3, 5 } };
    ttl::bind(y, i) = ttl::bind(A, i, j) * ttl::bind(x, j);
    assert(y[0] == A[0][0] * x[0] + A[0][1] * x[1]);
    assert(y[1] == A[1][0] * x[0] + A[1][1] * x[1]);

    return true;
}

static constexpr bool _tensors()
{
    int x[2]{2, 3};
    int y[2]{4, 5};
    int k[4]{};

    auto X = ttl::tspan(x);
    auto Y = ttl::tspan(y);
    auto K = ttl::tspan(k, 2, 2);
    K = X(i) * Y(j);
    assert((K[0,0] = X[0] * Y[0]));
    assert((K[1,0] = X[1] * Y[0]));
    assert((K[0,1] = X[0] * Y[1]));
    assert((K[1,1] = X[1] * Y[1]));

    // X += Y(i);

    return true;
}

int main()
{
    constexpr auto _ = _scalars();
    constexpr auto _ = _vectors();
    constexpr auto _ = _tensors();
    return 0;
}