#undef DNDEBUG

#include <ttl/ttl.hpp>
#include <cassert>
#include <array>
#include <mdspan>
#include <vector>

template <std::size_t... i>
static constexpr auto _extents = std::extents<std::size_t, i...>{i...};

static consteval auto check_scalars()
{
    int x = 0;
    assert(_extents<> == ttl::extents(x));
    assert(0 == ttl::evaluate(x));
    ttl::evaluate(x) = 1;
    assert(x == 1);
}

static consteval auto check_spans()
{
    int x[2]{0, 1};
    assert(_extents<2> == ttl::extents(x));
    assert(0 == ttl::evaluate(x, 0));
    assert(1 == ttl::evaluate(x, 1));
    ttl::evaluate(x, 0) = 1;
    ttl::evaluate(x, 1) = 2;
    assert(1 == x[0]);
    assert(2 == x[1]);

    int y[2][2]{{0, 1}, {2, 3}};
    assert((_extents<2, 2> == ttl::extents(y)));
    assert((0 == ttl::evaluate(y, 0, 0)));
    assert((1 == ttl::evaluate(y, 0, 1)));
    assert((2 == ttl::evaluate(y, 1, 0)));
    assert((3 == ttl::evaluate(y, 1, 1)));
    ttl::evaluate(y, 0, 0) = 1;
    ttl::evaluate(y, 0, 1) = 2;
    ttl::evaluate(y, 1, 0) = 3;
    ttl::evaluate(y, 1, 1) = 4;
    assert(1 == y[0][0]);
    assert(2 == y[0][1]);
    assert(3 == y[1][0]);
    assert(4 == y[1][1]);

    std::array z = {0, 1};
    assert(_extents<2> == ttl::extents(z));
    assert(0 == ttl::evaluate(z, 0));
    assert(1 == ttl::evaluate(z, 1));
    ttl::evaluate(z, 0) = 1;
    ttl::evaluate(z, 1) = 2;
    assert(1 == z[0]);
    assert(2 == z[1]);

    std::array w{std::array{0, 1}, std::array{2, 3}};
    assert((_extents<2, 2> == ttl::extents(w)));
    assert((0 == ttl::evaluate(w, 0, 0)));
    assert((1 == ttl::evaluate(w, 0, 1)));
    assert((2 == ttl::evaluate(w, 1, 0)));
    assert((3 == ttl::evaluate(w, 1, 1)));
    ttl::evaluate(w, 0, 0) = 1;
    ttl::evaluate(w, 0, 1) = 2;
    ttl::evaluate(w, 1, 0) = 3;
    ttl::evaluate(w, 1, 1) = 4;
    assert(1 == w[0][0]);
    assert(2 == w[0][1]);
    assert(3 == w[1][0]);
    assert(4 == w[1][1]);

    std::vector s{0, 1};
    assert(_extents<2> == ttl::extents(s));
    assert(0 == ttl::evaluate(s, 0));
    assert(1 == ttl::evaluate(s, 1));
    ttl::evaluate(s, 0) = 1;
    ttl::evaluate(s, 1) = 2;
    assert(1 == s[0]);
    assert(2 == s[1]);

    std::vector t{std::vector{0, 1}, std::vector{2, 3}};
    assert((_extents<2, 2> == ttl::extents(t)));
    assert((0 == ttl::evaluate(t, 0, 0)));
    assert((1 == ttl::evaluate(t, 0, 1)));
    assert((2 == ttl::evaluate(t, 1, 0)));
    assert((3 == ttl::evaluate(t, 1, 1)));
    ttl::evaluate(t, 0, 0) = 1;
    ttl::evaluate(t, 0, 1) = 2;
    ttl::evaluate(t, 1, 0) = 3;
    ttl::evaluate(t, 1, 1) = 4;
    assert(1 == t[0][0]);
    assert(2 == t[0][1]);
    assert(3 == t[1][0]);
    assert(4 == t[1][1]);
}

static consteval auto check_mdspans()
{
    int xʹ[2]{0, 1};
    auto x = std::mdspan(xʹ, 2);
    assert(_extents<2> == ttl::extents(x));
    assert(0 == ttl::evaluate(x, 0));
    assert(1 == ttl::evaluate(x, 1));
    ttl::evaluate(x, 0) = 1;
    ttl::evaluate(x, 1) = 2;
    assert(1 == x[0]);
    assert(2 == x[1]);

    int yʹ[4]{0, 1, 2, 3};
    auto y = std::mdspan(yʹ, 2, 2);
    assert((_extents<2, 2> == ttl::extents(y)));
    assert((0 == ttl::evaluate(y, 0, 0)));
    assert((1 == ttl::evaluate(y, 0, 1)));
    assert((2 == ttl::evaluate(y, 1, 0)));
    assert((3 == ttl::evaluate(y, 1, 1)));
    ttl::evaluate(y, 0, 0) = 1;
    ttl::evaluate(y, 0, 1) = 2;
    ttl::evaluate(y, 1, 0) = 3;
    ttl::evaluate(y, 1, 1) = 4;
    assert((1 == y[0,0]));
    assert((2 == y[0,1]));
    assert((3 == y[1,0]));
    assert((4 == y[1,1]));
}

static consteval void check_index()
{
    ttl::index_string i = "i";
    ttl::index_string j = "j";
    ttl::index_string k = "k";
    ttl::index_string ij = i + j;
    ttl::index_string ijk = ij + k;
    ttl::index_string ijj = ij + j;
    assert(ijk == ttl::index_string("ijk"));
    assert(3 == ijk.rank());
    assert(ijk == ijk.outer());
    assert(i == ijj.outer());
    assert(j == ijj.contracted());
    ttl::index_string s = "*";
    ttl::index_string isj = i + s + j;
    assert(ij == isj.outer());
    ttl::index_string isi = i + s + i;
    assert(i == isi.contracted());

    ttl::index<"i"> _;
    ttl::index<"j"> _;
    ttl::index<"ij"> _;

    ttl::index p = 1;
    assert(p._projected[0] == 1);
    ttl::index pp = p + p;
    assert(pp._projected[0] == 1);
    assert(pp._projected[1] == 1);

    using namespace ttl::literals;
    assert("abc"_id == "a"_id + "b"_id + "c"_id);
}

static consteval auto check_bind_scalar()
{
    int n = 0;
    auto b = ttl::bind(n);
    assert(0 == b[]);
    assert(0 == b);
    b[] = 1;
    assert(n == 1);
    b = 2.;
    assert(n == 2);

    auto c = ttl::bind(1);
    assert(1 == c[]);
    assert(1 == c);
}

static consteval auto check_bind_spans()
{
    ttl::index<"i"> i;
    int n[2]{0, 1};
    auto b = ttl::bind(n, i);
    assert(0 == b[0]);
    assert(1 == b[1]);
    b[0] = 1;
    b[1] = 2;
    assert(1 == n[0]);
    assert(2 == n[1]);

    int m[2][2]{ {1,0}, {0,1}};
    auto c = ttl::bind(m, i, i);
    static_assert(ttl::index_string{} == c.outer());
    assert(_extents<> == c.extents());
    static_assert(0 == ttl::rank<decltype(c)>);
    assert(2 == c[]);
    static_assert(ttl::scalar<decltype(c)>);
    assert(2 == c);

    ttl::index _0(1);
    ttl::index _(_0);
    ttl::index _(ttl::index(1));
    ttl::index _(i);

    auto d = ttl::bind(m, i, 1);
    assert(d._rank == 1);
    assert(0 == d[0]);
    assert(1 == d[1]);
    d[0] = 1;
    d[1] = 0;
    assert(1 == m[0][1]);
    assert(0 == m[1][1]);
}

static consteval auto check_tspan()
{
    ttl::index<"i"> i;
    ttl::index<"j"> j;

    int x[4]{0, 1, 2, 3};
    auto a = ttl::tspan(x, 2, 2);
    auto b = a(i, j);
    assert((0 == b[0,0]));
    assert((1 == b[0,1]));
    assert((2 == b[1,0]));
    assert((3 == b[1,1]));
    b[0,0] += 1;
    b[0,1] += 1;
    b[1,0] += 1;
    b[1,1] += 1;
    assert(1 == x[0]);
    assert(2 == x[1]);
    assert(3 == x[2]);
    assert(4 == x[3]);

    auto c = ttl::tspan(x, std::extents<std::size_t, 2, 2>{});
    auto d = a(i, j);
    assert((1 == d[0,0]));
    assert((2 == d[0,1]));
    assert((3 == d[1,0]));
    assert((4 == d[1,1]));
    d[0,0] += 1;
    d[0,1] += 1;
    d[1,0] += 1;
    d[1,1] += 1;
    assert(2 == x[0]);
    assert(3 == x[1]);
    assert(4 == x[2]);
    assert(5 == x[3]);

    auto e = std::mdspan(x, 2, 2);
    auto f = ttl::tspan(e);
    auto g = f;
    std::mdspan h = g;
    ttl::tspan s = h;
    auto t = s(i, j);
    assert((2 == t[0,0]));
    assert((3 == t[0,1]));
    assert((4 == t[1,0]));
    assert((5 == t[1,1]));
    t[0,0] += 1;
    t[0,1] += 1;
    t[1,0] += 1;
    t[1,1] += 1;
    assert(3 == x[0]);
    assert(4 == x[1]);
    assert(5 == x[2]);
    assert(6 == x[3]);

    int y = 0;
    std::mdspan m{&y};
    static_assert(ttl::expression<decltype(m)>);
    ttl::tspan n{m};
    static_assert(ttl::expression<decltype(n)>);
    auto o = n();
    assert(0 == o);
    o = 1;
    assert(1 == o);
}

static consteval auto check_product()
{
    ttl::index<"i"> i;
    ttl::index<"j"> j;

    int a = ttl::bind(2) * ttl::bind(2);
    assert(4 == a);

    int x[]{1, 2};
    auto xʹ = ttl::tspan(x, 2);
    int b = xʹ(i) * xʹ(i);
    assert(b == 1*1 + 2*2);

    auto k = xʹ(i) * xʹ(j);
    assert((k[0,0] == x[0]*x[0]));
    assert((k[0,1] == x[0]*x[1]));
    assert((k[1,0] == x[1]*x[0]));
    assert((k[1,1] == x[1]*x[1]));

    int A[]{1, 2, 4, 3};
    auto Aʹ = ttl::tspan(A, 2, 2);
    auto mv = Aʹ(i,j) * xʹ(j);
    assert((mv[0] == Aʹ[0,0] * xʹ[0] + Aʹ[0,1] * xʹ[1]));
    assert((mv[1] == Aʹ[1,0] * xʹ[0] + Aʹ[1,1] * xʹ[1]));

    int y[]{4, 2};
    auto yʹ = ttl::tspan(y, std::extents<std::size_t, 2>{});
    auto saxpy = 2 * Aʹ(i,j) * xʹ(j) + yʹ(i);
    assert(saxpy[0] == 2 * (Aʹ[0,0]*x[0] + Aʹ[0,1]*x[1]) + y[0]);
    assert(saxpy[1] == 2 * (Aʹ[1,0]*x[0] + Aʹ[1,1]*x[1]) + y[1]);
}

int main() {
    check_scalars();
    check_spans();
    check_mdspans();
    check_index();
    check_bind_scalar();
    check_bind_spans();
    check_tspan();
    check_product();
}
