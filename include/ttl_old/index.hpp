#pragma once

#include <ttl/index_string.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <ranges>
#include <utility>

namespace ttl
{
    template <index_string _str>
    struct index {
        std::size_t _projected[_str.size()] {};

        constexpr index() = default;

        constexpr index(std::size_t p)
            : _projected { p }
        {
            static constexpr char projection[] { projected_index, '\0' };
            static_assert(_str == index_string { projection });
        }

        constexpr bool operator==(index b) const
        {
            return std::ranges::equal(_projected, b._projected);
        }

        constexpr auto operator[](std::size_t i) const -> std::size_t
        {
            assert(i < _str.size());
            return _projected[i];
        }

        template <index_string other>
        constexpr auto operator+(index<other> const& b) const -> index<_str + other>
        {
            index<_str + other> out;
            std::ranges::copy(b._projected, std::ranges::copy(_projected, out._projected).out);
            return out;
        }

        constexpr auto projection_map() const
        {
            static constexpr auto n = _str.count(projected_index);
            static constexpr auto map = [] {
                std::array<std::size_t, n> out;
                int p = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    out[i] = _str.index_of(projected_index, p);
                }
                return out;
            }();
            return []<std::size_t... i>(std::index_sequence<i...>) {
                return std::index_sequence<map[i]...>();
            }(std::make_index_sequence<n>());
        }
    };

    index(std::size_t) -> index<index_string { "*" }>;

    inline namespace literals
    {
        template <index_string i>
        constexpr auto operator""_id() -> index<i>
        {
            return {};
        };
    }

    namespace _
    {
        template <class>
        struct is_index : std::false_type {
        };

        template <std::integral T>
        struct is_index<T> : std::true_type {
        };

        template <index_string str>
        struct is_index<index<str>> : std::true_type {
        };
    }

    template <class T>
    concept is_index = _::is_index<std::remove_cvref_t<T>>::value;
}
