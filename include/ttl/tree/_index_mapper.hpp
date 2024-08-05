#pragma once

#include <ttl/ARROW.hpp>
#include <ttl/concepts.hpp>
#include <ttl/sequence.hpp>
#include <ttl/tensor.hpp>
#include <ttl/tensor_index.hpp>
#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <ranges>

namespace ttl::tree
{
    template <std::size_t N>
    struct _index_mapper
    {
        // static constexpr std::size_t N = _to.size();

        int _map[N]{};

        template <std::size_t From, std::size_t To>
        constexpr _index_mapper(tensor_index<From> const& from, tensor_index<To> const& to)
        {
            assert(N == to.size());
            assert(to.is_subset_of(from));
            int i = 0;
            std::ranges::for_each(to, [&](auto const c) {
                _map[i++] = from.index_of(c);
            });

            // assert(_to.is_subset_of(from));
            // int i = 0;
            // std::ranges::for_each(_to, [&](auto const c) {
            //     _map[i++] = from.index_of(c);
            // });
        }

        template <concepts::tensor T, std::size_t M, std::size_t... i>
        constexpr auto _apply(T&& t, std::size_t const(&in)[M], sequence<i...>) const
            TTL_ARROW ( ttl::evaluate(std::forward<T>(t), in[_map[i]]...) );

        template <concepts::tensor T, concepts::size_t... Is>
        constexpr auto operator()(T&& t, Is... is) const
            TTL_ARROW ( _apply(std::forward<T>(t), {is...}, seqn<N>) );
    };
}
