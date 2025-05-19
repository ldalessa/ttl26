#pragma once

#include <ttl/tensor/tensor.hpp>
#include <ttl/tree/unary.hpp>

import std;

namespace ttl::tree
{
    template <concepts::expression A>
    struct identity : unary<A, std::identity {}> {
        using identity::unary::unary;
    };

	template <concepts::expression A>
    inline constexpr auto operator+(A&& a) -> identity<A> {
        return identity<A>(FWD(a));
    }
}
