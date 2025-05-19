#undef DNDEBUG

#include <ttl/ttl.hpp>

using namespace ttl::literals;

static constexpr auto i = "i"_id;
static constexpr auto j = "j"_id;

static constexpr bool _scalars()
{
    int a = 1, b = 2;
    int c = ttl::bind(a) + ttl::bind(b);
    assert(c == 3);

    return true;
}

static constexpr bool _vectors()
{
    int x[3]{1, 2, 3};
    int y[3]{3, 2, 1};
    auto xʹ = ttl::bind(x, i);
    auto yʹ = ttl::bind(y, i);
    static_assert(ttl::tensor<decltype(xʹ)>);
    static_assert(xʹ._rank == 1);
    static_assert(ttl::tensor<decltype(yʹ)>);
    static_assert(yʹ._rank == 1);
    auto z = xʹ + yʹ;
    assert(4 == z[0]);
    assert(4 == z[1]);
    assert(4 == z[2]);

    auto zʹ = xʹ - yʹ;
    assert(-2 == zʹ[0]);
    assert(0 == zʹ[1]);
    assert(2 == zʹ[2]);

    // auto q = x + ttl::bind(y, i);

    auto xy = ttl::bind(x, i) + ttl::bind(y, i);
    assert(4 == xy[0]);
    assert(4 == xy[1]);
    assert(4 == xy[2]);

    auto s = ttl::tspan(x, 3);
    auto t = ttl::tspan(y, 3);
    auto st = s(i) + t(i);
    assert(4 == st[0]);
    assert(4 == st[1]);
    assert(4 == st[2]);

    auto st2 = s(i) + t(i) + s(i) + t(i);
    assert(8 == st2[0]);
    assert(8 == st2[1]);
    assert(8 == st2[2]);

    return true;
}

static constexpr bool _tensors()
{
    int v[]{0, 1, 2, 0};
    int w[]{0, 2, 1, 0};
    auto vʹ = ttl::tspan(v, 2, 2);
    auto wʹ = ttl::tspan(w, 2, 2);
    auto vwʹ = vʹ(i,j) + wʹ(j,i);
    assert((0 == vwʹ[0,0]));
    assert((2 == vwʹ[0,1]));
    assert((4 == vwʹ[1,0]));
    assert((0 == vwʹ[1,1]));

    int m[]{0, 1};
    auto n = ttl::tspan(m, 2, 1);
    auto o = ttl::tspan(m, 1, 2);
    auto no = n(i,j) + o(j,i);
    assert((0 == no[0,0]));
    assert((2 == no[1,0]));

    return true;
}

int main()
{
    constexpr bool _ = _scalars();
    constexpr bool _ = _vectors();
    constexpr bool _ = _tensors();
    return 0;
}
