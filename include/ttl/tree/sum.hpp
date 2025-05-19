#pragma once
#include <ttl/concepts.hpp>
#include <ttl/index/imap.hpp>
#include <ttl/tensor/extents.hpp>
#include <ttl/tree/expression.hpp>
#include <cassert>
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
