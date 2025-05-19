#pragma once

#include <ttl/concepts.hpp>
#include <ttl/index/index.hpp>
#include <ttl/index/istring.hpp>
#include <ttl/tensor/tensor.hpp>
#include <ttl/tree/expression.hpp>
#include <cassert>
import std;

namespace ttl::tree
{
	namespace stdv = std::views;

	template <concepts::tensor A, istring _index>
	struct bind : expression
	{
		static_assert(rank<A> == _index.size());
		static_assert(_check_contracted_extents_static<_index, extents_type<A>>);

		using expression::operator=;
		
		using scalar_type = ttl::scalar_type<A>;

		static constexpr auto _outer = _index.outer();
		static constexpr auto _inner = _index.inner();
		static constexpr auto _all = _index.all();

		A _a;
		index<_index> _i{};

		template <istring... indices>
		constexpr bind(A a, index<indices>... is)
				: _a(a)
				, _i((index<"">{} + ... + is))
		{
			static_assert((istring{""} + ... + indices) == _index);
			assert(_check_contracted_extents_dynamic<_index>(ttl::extents(_a)));
		}

		template <class I, class... Is>
		requires (std::integral<I> or ... or std::integral<Is>)
		constexpr bind(A a, I i, Is... is)
				: bind(a, index(i), index(is)...)
		{
		}

		static constexpr auto rank = std::integral_constant<std::size_t, _outer.rank()>();

		static consteval auto outer() {
			return _outer;
		}

		constexpr auto extents() const
			-> ARROW( select_extents<_index, _outer>(ttl::extents(_a)) );

		/// Innermost evaluation just remaps indices
		constexpr auto operator[](this auto&& self, std::integral auto... i) -> decltype(auto)
			requires (sizeof...(i) == _all.size())
		{
			return FWD(self)._evaluate(imap<_all, _index>, i...);
		}

		/// Need to inject the projected indices.
		constexpr auto operator[](this auto&& self, std::integral auto... i) -> decltype(auto)
			requires (_inner.size() <= sizeof...(i) and sizeof...(i) < _all.size())
		{
			return FWD(self)._project(self._projection_map(), i...);
		}

		/// Need to contract indices.
		constexpr auto operator[](this auto&& self, std::integral auto... i) -> scalar_type
			requires (rank <= sizeof...(i) and sizeof...(i) < _inner.size())
		{
			static constexpr std::size_t N = sizeof...(i);
			auto const contracted = select_extents<_inner, _index>(ttl::extents(self._a));
			std::size_t const e = contracted.extent(N);
			scalar_type accum {};
			for (std::size_t j = 0; j < e; ++j) {
				accum += self[i..., j];
			}
			return accum;
		}

	  private:
		constexpr auto _projection_map() const {
			return _i.projection_map();
		}

		template <std::size_t... j>
		constexpr auto _evaluate(this auto&& self, std::index_sequence<j...>, std::integral auto... i) -> decltype(auto)
		{
			static_assert(sizeof...(i) == _all.size());
			static_assert(sizeof...(j) == _index.size());
			
			return evaluate(FWD(self)._a, i...[j]...);
		}

		template <std::size_t... j>
		constexpr auto _project(this auto&& self, std::index_sequence<j...>, std::integral auto... i) -> decltype(auto)
		{
			static_assert(sizeof...(i) == _inner.size());
			return FWD(self)[i..., self._i[j]...];
		}
	};

	template <concepts::expression T, istring... str>
	constexpr auto expression::_rebind(this T&& self, index<str>... is)
		-> decltype( bind(FWD(self), is...) )
	{
		// Make sure that this rebinding makes sense.
		static constexpr istring outer = ttl::outer<T>;
		static constexpr istring next = (istring{""} + ... + str);

		// We need one index for each slot.
		static_assert(outer.size() == next.size());

		// Any non-projected index, i, that appears uncontracted in next needs
		// to match the index in the same slot of outer.
		static_assert([] {
			for (auto const& [i, j] : stdv::zip(next, outer)) {
				if (i == j) continue;			   // match
				if (i == next.projected) continue; // projection
				if (0 == outer.count(i)) continue; // good rebind
				if (2 == next.count(i)) continue;  // contraction
				return false;
			}
			return true;
		}());

		return bind(FWD(self), is...);
	}
}
