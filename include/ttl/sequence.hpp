#pragma once

#include <cstddef>
#include <utility>

namespace ttl
{
    inline
    namespace _
    {
        template <std::size_t... vs>
        struct sequence {
            consteval sequence() = default;
            consteval sequence(std::index_sequence<vs...>) {}
        };

        template <auto N>
        inline constexpr sequence seqn = std::make_index_sequence<N>();

        template <std::size_t... vs>
        inline constexpr sequence seqv = seqn<sizeof...(vs)>;
    }
}
