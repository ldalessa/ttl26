#undef DNDEBUG

#include <ttl/tree/bind.hpp>
#include <ttl/tree/multiply.hpp>
#include <ttl/tree/product.hpp>

using namespace ttl;
using namespace ttl::tree;

static constexpr bool test_scalar()
{
	multiply<int, int> a(2, 3);
	assert(a == 6);

	auto b = bind(-1);
	auto c = b * 2;
	assert(c == -2);
	return true;
}

static_assert(test_scalar());

static constexpr bool test_vector_inner_product()
{
	static constexpr index<"i"> i;

	{
		int const x[] = {2};
		int const y[] = {3};

		auto a = bind(x, i);
		auto b = bind(y, i);

		multiply<decltype(a), decltype(b)> s(a, b);
		assert(s == 6);

		auto t = a * b;
		assert(t == 6);
	}

	{
		int const x[] = {2, 3};
		int const y[] = {3, 4};

		auto a = bind(x, i);
		auto b = bind(y, i);

		multiply<decltype(a), decltype(b)> s(a, b);
		assert(s == 18);

		auto t = a * b;
		assert(t == 18);
	}

	return true;
}

static_assert(test_vector_inner_product());

static constexpr bool test_matrix_vector_product()
{
	static constexpr index<"i"> i;
	static constexpr index<"j"> j;
	
	{
		int const x[2][3] = {{2, 3, 4}, {5, 6, 7}};
		int const y[] = {2, 2, 2};

		auto a = bind(x, i, j);
		auto b = bind(y, j);

		multiply<decltype(a), decltype(b)> s(a, b);
		static_assert(concepts::tensor_of_rank<decltype(s), 1zu>);
		static_assert(extent<0>(s) == 2zu);
		assert((s[0] == 18));
		assert((s[1] == 36));

		auto t = a * b;
		assert((t[0] == 18));
		assert((t[1] == 36));
	}

	{
		int const x[3][2] = {{2, 3}, {4, 5}, {6, 7}};
		int const y[] = {2, 2, 2};

		auto a = bind(x, j, i);
		auto b = bind(y, j);

		multiply<decltype(a), decltype(b)> s(a, b);
		static_assert(concepts::tensor_of_rank<decltype(s), 1zu>);
		static_assert(extent<0>(s) == 2zu);
		assert((s[0] == 24));
		assert((s[1] == 30));

		auto t = a * b;
		assert((t[0] == 24));
		assert((t[1] == 30));
 	}

	return true;
}

static_assert(test_matrix_vector_product());

static constexpr bool test_matrix_matrix_product()
{
	static constexpr index<"i"> i;
	static constexpr index<"j"> j;
	static constexpr index<"k"> k;
		
	{
		int const x[2][3] = {
			{2, 3, 4},
			{5, 6, 7}
		};
		
		int const y[3][2] = {
			{1, 2},
			{3, 4},
			{5, 6}
		};

		auto a = bind(x, i, j);
		auto b = bind(y, j, k);

		multiply<decltype(a), decltype(b)> s(a, b);
		static_assert(concepts::tensor_of_rank<decltype(s), 2zu>);
		static_assert(extent<0>(s) == 2zu);
		static_assert(extent<1>(s) == 2zu);
		assert((s[0,0] == 2*1 + 3*3 + 4*5));
		assert((s[0,1] == 2*2 + 3*4 + 4*6));
		assert((s[1,0] == 5*1 + 6*3 + 7*5));
		assert((s[1,1] == 5*2 + 6*4 + 7*6));		

		auto t = a * b;
		assert((t[0,0] == 2*1 + 3*3 + 4*5));
		assert((t[0,1] == 2*2 + 3*4 + 4*6));
		assert((t[1,0] == 5*1 + 6*3 + 7*5));
		assert((t[1,1] == 5*2 + 6*4 + 7*6));
	}

	{
		int const x[2][3] = {
			{2, 3, 4},
			{5, 6, 7}
		};
		
		int const y[3][2] = {
			{1, 2},
			{3, 4},
			{5, 6}
		};

		auto a = bind(x, i, j);
		auto b = bind(y, k, i);

		multiply<decltype(b), decltype(a)> s(b, a);
		static_assert(concepts::tensor_of_rank<decltype(s), 2zu>);
		static_assert(extent<0>(s) == 3zu);
		static_assert(extent<1>(s) == 3zu);
		assert((s[0,0] == 1*2 + 2*5));
		assert((s[0,1] == 1*3 + 2*6));
		assert((s[0,2] == 1*4 + 2*7));

		assert((s[1,0] == 3*2 + 4*5));
		assert((s[1,1] == 3*3 + 4*6));
		assert((s[1,2] == 3*4 + 4*7));

		assert((s[2,0] == 5*2 + 6*5));
		assert((s[2,1] == 5*3 + 6*6));
		assert((s[2,2] == 5*4 + 6*7));

		auto t = a * b;
		assert((t[0,0] == 1*2 + 2*5));
		assert((t[0,1] == 1*3 + 2*6));
		assert((t[0,2] == 1*4 + 2*7));

		assert((t[1,0] == 3*2 + 4*5));
		assert((t[1,1] == 3*3 + 4*6));
		assert((t[1,2] == 3*4 + 4*7));

		assert((t[2,0] == 5*2 + 6*5));
		assert((t[2,1] == 5*3 + 6*6));
		assert((t[2,2] == 5*4 + 6*7));
	}
	
	return true;
}

static_assert(test_matrix_vector_product());
