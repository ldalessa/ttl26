#undef DNDEBUG

#include <ttl/tspan.hpp>
#include <ttl/index/index.hpp>
#include <ttl/tree/bind.hpp>
#include <ttl/tree/subtract.hpp>

using namespace ttl;
using namespace ttl::tree;

static_assert(concepts::tensor<subtract<int, int>>);
static_assert(concepts::expression<subtract<int, int>>);

static constexpr bool test_scalar_sub()
{
	subtract<int, int> a(1, 1);
	assert(a[] == 0);
	return true;
}

static constexpr bool test_vector_sub()
{
	static constexpr index<"i"> i;

	{
		int const x[] = {1};
		int const y[] = {1};

		auto a = bind(x, i);
		auto b = bind(y, i);

		subtract<decltype(a), decltype(b)> s(a, b);
		assert(s[0] == 0);

		auto t = a - b;
		assert(t[0] == 0);
	}

	{
		int const x[] = {1, 2};
		int const y[] = {1, 2};

		auto a = bind(x, i);
		auto b = bind(y, i);

		subtract<decltype(a), decltype(b)> s(a, b);
		assert(s[0] == 0);
		assert(s[1] == 0);

		auto t = a - b;
		assert(t[0] == 0);
		assert(t[1] == 0);
	}

	return true;
}

static constexpr bool test_matrix_sub()
{
	static constexpr index<"i"> i;
	static constexpr index<"j"> j;

	int const x[2][2] = {{1, 2}, {3, 4}};
	int const y[2][2] = {{1, 2}, {3, 4}};

	{
		auto a = bind(x, i, j);
		auto b = bind(y, i, j);

		subtract<decltype(a), decltype(b)> s(a, b);
		assert((s[0,0] == 0));
		assert((s[0,1] == 0));
		assert((s[1,0] == 0));
		assert((s[1,1] == 0));

		auto t = a - b;
		assert((t[0,0] == 0));
		assert((t[0,1] == 0));
		assert((t[1,0] == 0));
		assert((t[1,1] == 0));
	}

	{
		auto a = bind(x, i, j);
		auto b = bind(y, j, i);

		subtract<decltype(a), decltype(b)> s(a, b);
		assert((s[0,0] == 0));
		assert((s[0,1] == -1));
		assert((s[1,0] == 1));
		assert((s[1,1] == 0));

		auto t = a - b;
		assert((t[0,0] == 0));
		assert((t[0,1] == -1));
		assert((t[1,0] == 1));
		assert((t[1,1] == 0));
	}

	return true;
}

static_assert(test_scalar_sub());
static_assert(test_vector_sub());
static_assert(test_matrix_sub());
