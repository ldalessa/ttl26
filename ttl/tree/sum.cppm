module;
#include <cassert>
export module ttl:sum;
export import :bind;							// for unit testing
export import :concepts;
export import :expression;
export import :extents;
export import :imap;
export import :index;							// for unit testing
import std;

namespace ttl::tree
{
	/// A generic sum node, will apply op(A, B).
	///
	/// The sum supports any sort of restructuring of indices as long as the
	/// extents match. For example A(i,j) + A(j,i) represents A + A^T.
	template <concepts::expression A, concepts::expression B, auto op>
	struct sum : expression
	{
		using scalar_type = std::invoke_result_t<decltype(op), ttl::scalar_type<A>, ttl::scalar_type<B>>;

		static constexpr auto _outer_a = ttl::outer<A>;
		static constexpr auto _outer_b = ttl::outer<B>;
		static_assert(is_permutation(_outer_a, _outer_b));

		static constexpr auto _map_aa = imap<_outer_a, _outer_a>;
		static constexpr auto _map_ab = imap<_outer_a, _outer_b>;
		static constexpr auto _map_ba = imap<_outer_b, _outer_a>;

		static constexpr auto rank = std::integral_constant<std::size_t, _outer_a.rank()>();

		A _a;
		B _b;

		constexpr sum(A a, B b)
			: _a(FWD(a))
			, _b(FWD(b))
		{
			auto const extents_a = select_extents(_map_aa, ttl::extents(_a));
			auto const extents_b = select_extents(_map_ba, ttl::extents(_b));
			assert(compatible_extents(extents_a, extents_b));
		}

		static consteval auto outer() {
			return _outer_a;
		}

		constexpr auto extents() const
		{
			auto const extents_a = select_extents(_map_aa, ttl::extents(_a));
			auto const extents_b = select_extents(_map_ba, ttl::extents(_b));
			return merge_extents(extents_a, extents_b);
		}

		constexpr auto operator[](std::integral auto... i) const -> scalar_type
		{
			static_assert(sizeof...(i) == rank);
			assert(_check_bounds(i...));
			scalar_type a = _evaluate(_a, _map_aa, i...);
			scalar_type b = _evaluate(_b, _map_ab, i...);
			return op(std::move(a), std::move(b));
		}

	private:
		template <std::size_t... i>
		static constexpr auto _evaluate(auto&& x, std::index_sequence<i...>, std::integral auto... j)
			-> ARROW( ttl::evaluate(FWD(x), j...[i]...) );
	};

	template <concepts::expression A, concepts::expression B>
	struct add : sum<A, B, std::plus{}> {
		using add::sum::sum;
	};

	template <concepts::expression A, concepts::expression B>
	struct sub : sum<A, B, std::minus{}> {
		   using sub::sum::sum;
	};

	template <concepts::expression A, concepts::expression B>
	inline constexpr auto operator+(A&& a, B&& b) -> add<A, B>
	{
		   return add<A, B>(FWD(a), FWD(b));
	}

	template <concepts::expression A, concepts::expression B>
	inline constexpr auto operator-(A&& a, B&& b) -> sub<A, B>
	{
		   return sub<A, B>(FWD(a), FWD(b));
	}
}

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
