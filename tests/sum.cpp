#undef DNDEBUG

#include <ttl/index/index.hpp>
#include <ttl/tree/bind.hpp>
#include <ttl/tree/sum.hpp>

using namespace ttl;
using namespace ttl::tree;

static_assert(concepts::tensor<add<int, int>>);
static_assert(concepts::expression<add<int, int>>);

static_assert(concepts::tensor<sub<int, int>>);
static_assert(concepts::expression<sub<int, int>>);

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

static constexpr bool test_scalar_sub()
{
	sub<int, int> a(1, 1);
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

		sub<decltype(a), decltype(b)> s(a, b);
		assert(s[0] == 0);

		auto t = a - b;
		assert(t[0] == 0);
	}

	{
		int const x[] = {1, 2};
		int const y[] = {1, 2};

		auto a = bind(x, i);
		auto b = bind(y, i);

		sub<decltype(a), decltype(b)> s(a, b);
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

		sub<decltype(a), decltype(b)> s(a, b);
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

		sub<decltype(a), decltype(b)> s(a, b);
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

static_assert(test_scalar_add());
static_assert(test_vector_add());
static_assert(test_matrix_add());

static_assert(test_scalar_sub());
static_assert(test_vector_sub());
static_assert(test_matrix_sub());

