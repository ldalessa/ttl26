#pragma once

#include <ttl/expression.hpp>
#include <ttl/extents.hpp>
#include <ttl/evaluate.hpp>
#include <ttl/outer.hpp>
#include <ttl/tree/node.hpp>

#include <concepts>
#include <functional>

namespace ttl::tree
{
    template <expression A, auto op>
    struct unary_prefix : node
    {
        A _a;

        constexpr operator ttl::evaluate_type<A>(this auto&& self)
            requires (ttl::rank<A> == 0)
        {
            return __fwd(self)[];
        }

        static constexpr auto outer() {
            return ttl::outer<A>;
        }

        constexpr auto extents() const {
            return ttl::extents(_a);
        }

        constexpr auto operator[](this auto&& self, std::integral auto... i) -> ttl::evaluate_type<A>
        {
            static_assert(sizeof...(i) == ttl::rank<A>);
            assert(self._check_bounds(i...));
            return op(ttl::evaluate(__fwd(self)._a, i...));
        }
    };

    template <expression A>
    struct negate : unary_prefix<A, std::negate{}> {
        using negate::unary_prefix::unary_prefix;
    };

    template <expression A>
    struct identity : unary_prefix<A, std::identity{}> {
        using identity::unary_prefix::unary_prefix;
    };

    template <expression A>
    inline constexpr auto operator-(A&& a) -> negate<A> {
        return negate<A>(__fwd(a));
    }

    template <expression A>
    inline constexpr auto operator+(A&& a) -> identity<A> {
        return identity<A>(__fwd(a));
    }
}
