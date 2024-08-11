#pragma once

#include <ttl/index.hpp>
#include <ttl/index_string.hpp>
#include <ttl/tensor.hpp>
#include <ttl/tree/bind.hpp>
#include <ttl/tree/node.hpp>

namespace ttl
{
    template <scalar A>
    inline constexpr auto bind(A& a) -> tree::bind<A&, ""> {
        return tree::bind<A&, "">(a);
    }

    template <scalar A>
    inline constexpr auto bind(A const&& a) -> tree::bind<A const, ""> {
        return tree::bind<A const, "">(std::move(a));
    }

    template <tensor A, index_string... _index>
    inline constexpr auto bind(A& a, index<_index> const&... ids)
        -> tree::bind<A&, (_index + ...)>
    {
        return tree::bind<A&, (_index + ...)>(a, (ids + ...));
    }

    template <tensor A, index_string... _index>
    inline constexpr auto bind(A const&& a, index<_index> const&... ids)
        -> tree::bind<A const, (_index + ...)>
    {
        return tree::bind<A const, (_index + ...)>(std::move(a), (ids + ...));
    }

    template <tensor A, class... Index>
    requires (std::integral<Index> or ...)
    inline constexpr auto bind(A&& a, Index const&... ids)
        // -> decltype(bind((A&&)a, ttl::index(ids)...))
        // https://github.com/llvm/llvm-project/issues/54440
    {
        return bind((A&&)a, ttl::index(ids)...);
    }
}

template <class T, ttl::index_string... str>
constexpr auto ttl::tree::node::operator()(this T&& self, index<str>... is)
    -> bind<T, (str + ...)>
{
    return ttl::bind(__fwd(self), is...);
}
