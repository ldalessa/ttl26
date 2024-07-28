#pragma once

#include <ttl/ARROW.hpp>
#include <ttl/evaluate.hpp>
#include <ttl/extents.hpp>
#include <ttl/rank.hpp>
#include <ttl/traits.hpp>
#include <array>
#include <concepts>
#include <cstddef>
#include <mdspan>
#include <span>
#include <vector>

namespace ttl
{
    namespace concepts
    {
        template <class T>
        concept tensor = has_rank<T> and has_evaluate<T, rank<T>> and has_extents<T>;

        template <class T, std::size_t N>
        concept tensor_of_rank = tensor<T> and rank<T> == N;

        template <class T>
        concept scalar = tensor_of_rank<T, 0zu>;
    }

    template <class T>
    requires (std::integral<T> or std::floating_point<T>)
    struct traits::tensor<T>
    {
        static consteval auto extents(T const&) -> std::extents<std::size_t> {
            return {};
        }

        static consteval auto rank() -> std::size_t {
            return 0zu;
        }

        static constexpr auto evaluate(T const& x) -> T const& {
            return x;
        }

        static constexpr auto evaluate(T& x) -> T& {
            return x;
        }
    };

    template <concepts::tensor T, std::size_t N>
    struct traits::tensor<std::span<T, N>>
    {
        static consteval auto extents(std::span<T, N> x)
            TTL_ARROW ( _::prepend<N>(ttl::extents(x[0])) );

        static consteval auto rank() -> std::size_t {
            return ttl::rank<T> + 1;
        }

        static constexpr auto evaluate(std::span<T, N> x, std::size_t i, std::integral auto... j)
            TTL_ARROW ( ttl::evaluate(x[i], j...) );

        /// Forwarding version is used for subclass rvalues that can't bind
        /// directly to a span type.
        template <class U>
        requires (not std::convertible_to<U, std::span<T, N>>)
        static constexpr auto evaluate(U&& x, std::integral auto... j)
            TTL_ARROW ( tensor::evaluate(std::span(x), j...) );

        template <class U>
        requires (not std::convertible_to<U, std::span<T, N>>)
        static constexpr auto extents(U&& x)
            TTL_ARROW ( tensor::extents(std::span(x)) );
    };

    template <concepts::tensor T, std::size_t N>
    struct traits::tensor<T[N]> : traits::tensor<std::span<T, N>>
    {
    };

    template <concepts::tensor T, std::size_t N>
    struct traits::tensor<T const[N]> : traits::tensor<std::span<T const, N>>
    {
    };

    template <concepts::tensor T, std::size_t N>
    struct traits::tensor<std::array<T, N>> : traits::tensor<std::span<T, N>>
    {
    };

    template <concepts::tensor T, std::size_t N>
    struct traits::tensor<const std::array<T, N>> : traits::tensor<std::span<T const, N>>
    {
    };

    template <concepts::tensor T>
    struct traits::tensor<std::vector<T>> : traits::tensor<std::span<T, std::dynamic_extent>>
    {
    };

    template <concepts::tensor T>
    struct traits::tensor<const std::vector<T>> : traits::tensor<std::span<T const>>
    {
    };
}
