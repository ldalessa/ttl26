#undef DNDEBUG

#include <ttl/tspan.hpp>
#include <ttl/index/index.hpp>
#include <ttl/tree/add.hpp>
#include <ttl/tree/bind.hpp>

using namespace ttl;
using namespace ttl::literals;
using namespace ttl::tree;

static_assert(concepts::tensor<add<int, int>>);
static_assert(concepts::expression<add<int, int>>);

static constexpr bool test_scalar_add()
{
	add<int, int> a(1, 1);
	assert(a[] == 2);

	{
		auto a = bind(1);
		auto b = a + 1;
		assert(b == 2);
	}

	return true;
}

static_assert(test_scalar_add());

static constexpr bool test_vector_add()
{
	static constexpr index<"i"> i;

	{
		int const x[] = {1};
		int const y[] = {1};

		auto a = bind(x, i);
		auto b = bind(y, i);

		add<decltype(a), decltype(b)> s(a, b);
		assert(s[0] == 2);

		auto t = a + b;
		assert(t[0] == 2);
	}

	{
		int const x[] = {1, 2};
		int const y[] = {1, 2};

		auto a = bind(x, i);
		auto b = bind(y, i);

		add<decltype(a), decltype(b)> s(a, b);
		assert(s[0] == 2);
		assert(s[1] == 4);

		auto t = a + b;
		assert(t[0] == 2);
		assert(t[1] == 4);
	}

	return true;
}

static_assert(test_vector_add());

static constexpr bool test_matrix_add()
{
	static constexpr index<"i"> i;
	static constexpr index<"j"> j;

	int const x[2][2] = {{1, 2}, {3, 4}};
	int const y[2][2] = {{1, 2}, {3, 4}};

	{
		auto a = bind(x, i, j);
		auto b = bind(y, i, j);

		add<decltype(a), decltype(b)> s(a, b);
		assert((s[0,0] == 2));
		assert((s[0,1] == 4));
		assert((s[1,0] == 6));
		assert((s[1,1] == 8));

		auto t = a + b;
		assert((t[0,0] == 2));
		assert((t[0,1] == 4));
		assert((t[1,0] == 6));
		assert((t[1,1] == 8));
	}

	{
		auto a = bind(x, i, j);
		auto b = bind(y, j, i);

		add<decltype(a), decltype(b)> s(a, b);
		assert((s[0,0] == 2));
		assert((s[0,1] == 5));
		assert((s[1,0] == 5));
		assert((s[1,1] == 8));

		auto t = a + b;
		assert((t[0,0] == 2));
		assert((t[0,1] == 5));
		assert((t[1,0] == 5));
		assert((t[1,1] == 8));
	}

	return true;
}

static_assert(test_matrix_add());

static constexpr bool _vectors()
{
	static constexpr index<"i"> i;

    int x[3]{1, 2, 3};
    int y[3]{3, 2, 1};
    auto xʹ = bind(x, i);
    auto yʹ = bind(y, i);
    static_assert(concepts::tensor<decltype(xʹ)>);
    static_assert(xʹ.rank == 1);
    static_assert(concepts::tensor<decltype(yʹ)>);
    static_assert(yʹ.rank == 1);
    auto z = xʹ + yʹ;
    assert(4 == z[0]);
    assert(4 == z[1]);
    assert(4 == z[2]);

    auto xy = bind(x, i) + bind(y, i);
    assert(4 == xy[0]);
    assert(4 == xy[1]);
    assert(4 == xy[2]);

    auto s = tspan(x, 3);
    auto t = tspan(y, 3);
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

static_assert(_vectors());

static constexpr bool _tensors()
{
	static constexpr index<"i"> i;
	static constexpr index<"j"> j;

    int v[]{0, 1, 2, 0};
    int w[]{0, 2, 1, 0};
    auto vʹ = tspan(v, 2, 2);
    auto wʹ = tspan(w, 2, 2);
    auto vwʹ = vʹ(i,j) + wʹ(j,i);
    assert((0 == vwʹ[0,0]));
    assert((2 == vwʹ[0,1]));
    assert((4 == vwʹ[1,0]));
    assert((0 == vwʹ[1,1]));

    int m[]{0, 1};
    auto n = tspan(m, 2, 1);
    auto o = tspan(m, 1, 2);
    auto no = n(i,j) + o(j,i);
    assert((0 == no[0,0]));
    assert((2 == no[1,0]));

    return true;
}

static_assert(_tensors());
