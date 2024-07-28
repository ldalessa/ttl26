#pragma once

#include <ttl/ARROW.hpp>
#include <ttl/traits.hpp>
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <mdspan>
#include <utility>

namespace ttl
{
    namespace concepts
    {
        template <class T>
        concept extents = requires (T t) {
            []<class U, std::size_t... us>(std::extents<U, us...>&) {}(t);
        };
    }

    inline constexpr struct _extents_fn
    {
      private:
        template <class T>
        struct _traits {
            static constexpr bool use_tensor_trait = requires (T&& t) {
                { auto(traits::tensor<T>::extents(std::forward<T>(t))) } -> concepts::extents;
            };

            static constexpr bool use_member_function = not use_tensor_trait and requires (T&& t) {
                { auto(std::forward<T>(t).extents()) } -> concepts::extents;
            };
        };

      public:
        template <class T>
        requires _traits<T>::use_tensor_trait
        static constexpr auto operator()(T&& t)
            TTL_ARROW ( traits::tensor<T>::extents(std::forward<T>(t)) );

        template <class T>
        requires _traits<T>::use_member_function
        static constexpr auto operator()(T&& t)
            TTL_ARROW ( std::forward<T>(t).extents() );

    } extents;

    namespace concepts
    {
        template <class T>
        concept has_extents = requires (T&& t) {
            { auto(ttl::extents(std::forward<T>(t))) } -> extents;
        };
    }

    namespace _
    {
        template <class T, std::size_t... as, std::size_t... bs>
        inline constexpr auto concat(std::extents<T, as...> const& a, std::extents<T, bs...> const& b) -> std::extents<T, as..., bs...> {
            return [&]<std::size_t... i, std::size_t... j>(std::index_sequence<i...>, std::index_sequence<j...>) {
                return std::extents<T, as..., bs...> { a.extent(i)..., b.extent(j)... };
            }(std::make_index_sequence<sizeof...(as)>(), std::make_index_sequence<sizeof...(bs)>());
        }

        template <std::size_t a, class T, std::size_t... bs>
        inline constexpr auto prepend(std::extents<T, bs...> const& b)
            TTL_ARROW( concat(std::extents<T, a>(static_cast<T>(a)), b) );

        template <std::size_t b, class T, std::size_t... as>
        inline constexpr auto append(std::extents<T, as...> const& a)
            TTL_ARROW ( concat(a,  std::extents<T, b>(static_cast<T>(b))) );

        template <std::integral auto... i, class T, std::size_t... ts>
        inline constexpr auto select(std::extents<T, ts...> const& t)
            TTL_ARROW ( std::extents<T, std::extents<T, ts...>::static_extent(i)...> { t.extents(i)... } );

        template <std::size_t... i, class T, std::size_t... ts>
        inline constexpr auto select(std::index_sequence<i...>, std::extents<T, ts...> const& t)
            TTL_ARROW ( select<i...>(t) );

        template <class T, std::size_t... a, std::size_t... b>
        inline constexpr bool similar(std::extents<T, a...> const&, std::extents<T, b...> const&)
        {
            if constexpr (sizeof...(a) != sizeof...(b)) {
                return false;
            }
            else {
                return ((a == b or a == std::dynamic_extent or b == std::dynamic_extent) && ...);
            }
        }

        template <class T, std::size_t... as, std::size_t... bs>
        inline constexpr auto merge(std::extents<T, as...> const& a, std::extents<T, bs...> const& b)
            -> std::extents<T, std::min(as, bs)...>
        {
            static_assert(sizeof...(as) == sizeof...(bs));
            static_assert(((as == bs or as == std::dynamic_extent or bs == std::dynamic_extent) && ...));

            [&]<std::size_t... i>(std::index_sequence<i...>) {
                (assert(a.extent(i) == b.extent(i)), ...);
            }(std::make_index_sequence<sizeof...(as)>());

            return std::extents<T, std::min(as, bs)...>{a};
        }
    }
}
