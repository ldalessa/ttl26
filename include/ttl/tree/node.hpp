#pragma once

#include <ttl/extents.hpp>
#include <ttl/index.hpp>

#include <concepts>
#include <cstddef>
#include <mdspan>
#include <utility>

namespace ttl::tree
{
    template <tensor, index_string>
    struct bind;

    struct node {
        /// Rebind an expression. This is implemented in bind.hpp in order to
        /// break the circular include there.
        template <class T, index_string... str>
        constexpr auto operator()(this T&& self, index<str>... is)
            -> bind<T, (str + ...)>;

        /// Rebind an expression.
        template <class T>
        constexpr auto operator()(this T&& self, is_index auto... i)
            -> decltype(__fwd(self)[index(i)...])
        {
            static_assert(sizeof...(i) == ttl::rank<T>);
            return __fwd(self)[index(i)...];
        }

        /// Allow assignments of scalar expressions to automatically evaluate
        /// their results.
        template <class T>
        constexpr operator decltype(std::declval<T&&>()[])(this T&& self)
            requires requires { std::declval<T&&>()[]; }
        {
            return __fwd(self)[];
        }

    protected:
        /// Check that the bounds are inside the extents.
        constexpr bool _check_bounds(this auto const& self, std::integral auto... i)
        {
            return [&]<std::size_t... n>(std::index_sequence<n...>) -> bool {
                return ((0 <= i and std::size_t(i) < self.extents().extent(n)) && ...);
            }(std::make_index_sequence<sizeof...(i)>());
        }

        /// Check that contracted indices have the same extents.
        template <index_string index, std::size_t... es>
        static constexpr bool _check_contracted_extents(std::extents<std::size_t, es...> const& extents)
        {
            // For each contracted index, check to make sure that the static
            // extents we are contracting are compatible and that the actual
            // extents are the same.
            static constexpr auto contracted = index.contracted();
            static constexpr std::size_t static_extents[]{es...}; // @todo[c++26]
            return [&]<std::size_t... i>(std::index_sequence<i...>) {
                return ([&] {
                    constexpr auto c = contracted[i];
                    constexpr auto x = index.find_offsets(c);
                    constexpr auto j = x[0];
                    constexpr auto k = x[1];
                    constexpr auto a = static_extents[j]; // @todo[c++26] es...[j];
                    constexpr auto b = static_extents[k]; // @todo[c++26] es...[k];
                    static_assert(a == b or a == std::dynamic_extent or b == std::dynamic_extent);
                    return extents.extent(j) == extents.extent(k);
                }() && ...);
            }(std::make_index_sequence<contracted.size()>());
        }
    };
}
