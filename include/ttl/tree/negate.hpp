#pragma once

#include <ttl/tensor/tensor.hpp>
#include <ttl/tree/unary.hpp>

import std;

namespace ttl::tree
{
    template <concepts::expression A>
    struct negate : unary<A, std::negate {}> {
        using negate::unary::unary;
    };

	template <concepts::expression A>
    inline constexpr auto operator-(A&& a) -> negate<A> {
        return negate<A>(FWD(a));
    }
}
