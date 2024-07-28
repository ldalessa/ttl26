#pragma once

#include <ttl/ARROW.hpp>
#include <ttl/evaluate.hpp>
#include <ttl/index.hpp>
#include <ttl/tensor.hpp>
#include <algorithm>
#include <cassert>
#include <utility>

namespace ttl
{
    namespace _
    {
        template <std::size_t M>
        struct index_mapper
        {
            int _map[M]{};

            template <std::size_t N>
            constexpr index_mapper(index<N> const& from, index<M> const& to)
            {
                assert(to.is_subset_of(from.outer()));
                std::for_each(to, [&,i=0](auto const c) mutable {
                    _map[i++] = from.index_of(c);
                });
            }

            template <concepts::tensor T>
            constexpr auto operator()(T&& t, std::integral auto... is) const
                TTL_ARROW ( _apply(std::forward<T>(t), {is...}, std::make_index_sequence<M>()) );

          private:
            template <concepts::tensor T, std::size_t N, std::size_t... i>
            constexpr auto _apply(T&& t, std::array<int, N> in, std::index_sequence<i...>) const
                TTL_ARROW ( ttl::evaluate(std::forward<T>(t), in[_map[i]]...) );
        };
    }
}
