#pragma once

#include <ttl/tensor/extents.hpp>
#include <ttl/tensor/evaluate.hpp>
#include <ttl/tensor/outer.hpp>
#include <ttl/tensor/rank.hpp>
#include <ttl/tensor/tensor.hpp>
#include <ttl/tree/expression.hpp>

import std;

namespace ttl::tree
{
    template <concepts::expression A, auto op>
    struct unary : expression
	{
		using scalar_type = ttl::scalar_type<A>;

		static constexpr auto rank = std::integral_constant<std::size_t, ttl::rank<A>>();
		
        A _a;

		constexpr unary(A a) : _a(FWD(a)) {}
		
        static constexpr auto outer() {
            return ttl::outer<A>;
        }

        constexpr auto extents() const {
            return ttl::extents(_a);
        }

        constexpr auto operator[](this auto&& self, std::integral auto... i) -> decltype(auto)
        {
            static_assert(sizeof...(i) == rank);
            assert(self._check_bounds(i...));
            return op(ttl::evaluate(FWD(self)._a, i...));
        }
    };
}
