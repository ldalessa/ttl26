#pragma once

#include <ttl/ARROW.hpp>
#include <ttl/concepts.hpp>
#include <ttl/sequence.hpp>
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <mdspan>
#include <type_traits>
#include <utility>

namespace ttl
{
    inline
    namespace _
    {
        template <class T, std::size_t... as, class U, std::size_t... bs>
        inline constexpr auto concat_extents(std::extents<T, as...> const& a, std::extents<U, bs...> const& b) -> std::extents<std::common_type_t<T>, as..., bs...> {
            return [&]<std::size_t... i, std::size_t... j>(sequence<i...>, sequence<j...>) {
                return std::extents<std::common_type_t<T, U>, as..., bs...> { a.extent(i)..., b.extent(j)... };
            }(seqv<as...>, seqv<bs...>);
        }

        template <std::size_t a, class T, std::size_t... bs>
        inline constexpr auto prepend_extent(std::extents<T, bs...> const& b, T x = a)
            TTL_ARROW( concat_extents(std::extents<T, a>(x), b) );

        template <std::size_t b, class T, std::size_t... as>
        inline constexpr auto append_extent(std::extents<T, as...> const& a, T x = b)
            TTL_ARROW ( concat_extents(a,  std::extents<T, b>(x)) );

        template <std::size_t... i, class T, std::size_t... ts>
        inline constexpr auto select_extents(std::extents<T, ts...> const& t)
            TTL_ARROW ( std::extents<T, std::extents<T, ts...>::static_extent(i)...> { t.extent(i)... } );

        template <std::size_t... i, class T, std::size_t... ts>
        inline constexpr auto select_extents(sequence<i...> const&, std::extents<T, ts...> const& t)
            TTL_ARROW ( select_extents<i...>(t) );

        template <class T, std::size_t... a, std::size_t... b>
        inline constexpr bool similar_extents(std::extents<T, a...> const&, std::extents<T, b...> const&)
        {
            if constexpr (sizeof...(a) != sizeof...(b)) {
                return false;
            }
            else {
                return ((a == b or a == std::dynamic_extent or b == std::dynamic_extent) && ...);
            }
        }

        template <class T, std::size_t... as, std::size_t... bs>
        inline constexpr auto merge_extents(std::extents<T, as...> const& a, std::extents<T, bs...> const& b)
            -> std::extents<T, std::min(as, bs)...>
        {
            static_assert(sizeof...(as) == sizeof...(bs));
            static_assert(((as == bs or as == std::dynamic_extent or bs == std::dynamic_extent) && ...));

            [&]<std::size_t... i>(sequence<i...>) {
                (assert(a.extent(i) == b.extent(i)), ...);
            }(seqv<as...>);

            return std::extents<T, std::min(as, bs)...>{a};
        }
    }
}
