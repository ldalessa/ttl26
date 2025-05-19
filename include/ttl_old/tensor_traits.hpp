#pragma once

#include <ttl/evaluate.hpp>
#include <ttl/extents.hpp>
#include <ttl/tensor.hpp>

#include <concepts>
#include <mdspan>
#include <span>

namespace ttl
{
    template <class>
    struct tensor_traits {
    }; // definining this produces better errors

    template <class T>
    struct tensor_traits<T const> : tensor_traits<T> {
    };

    template <class T>
        requires std::integral<T> or std::floating_point<T>
    struct tensor_traits<T> {
        static constexpr auto extents(T const&) -> std::extents<std::size_t>
        {
            return {};
        }

        static constexpr auto evaluate(T& t) -> T&
        {
            return t;
        }

        static constexpr auto evaluate(T const& t) -> T const&
        {
            return t;
        }
    };

    template <class T, std::size_t N>
    struct tensor_traits<std::span<T, N>> {
        static_assert(tensor<T>, "Spans must wrap tensor types.");

        using span = std::span<T, N>;

        static constexpr auto extents(span s)
        {
            return prepend_extent<N>(ttl::extents(s[0]), s.size());
        }

        template <std::integral... J>
        static constexpr auto evaluate(span s, std::size_t i, J... j)
            -> decltype(ttl::evaluate(s[i], j...))
        {
            return ttl::evaluate(s[i], j...);
        }
    };

    template <class T>
        requires requires(T&& t) {
            std::span(t);
        }
    struct tensor_traits<T> {
        static constexpr auto extents(T const& t)
        {
            return ttl::extents(std::span(t));
        }

        template <std::integral... J>
        static constexpr auto evaluate(T& t, std::size_t i, J... j)
            -> decltype(ttl::evaluate(std::span(t), i, j...))
        {
            return ttl::evaluate(std::span(t), i, j...);
        }

        template <std::integral... J>
        static constexpr auto evaluate(T const& t, std::size_t i, J... j)
            -> decltype(ttl::evaluate(std::span(t), i, j...))
        {
            return ttl::evaluate(std::span(t), i, j...);
        }
    };
}
