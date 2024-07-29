#pragma once

#include <ttl/extents.hpp>
#include <ttl/tensor_index.hpp>
#include <ttl/traits.hpp>
#include <array>
#include <concepts>
#include <mdspan>
#include <span>
#include <vector>

namespace ttl
{
    /// Scalar traits
    template <class T>
    requires (std::integral<T> or std::floating_point<T>)
    struct traits<T>
    {
        // [optional]
        using extents_type = std::extents<std::size_t>;

        // [required]
        static consteval auto extents(T) -> extents_type {
            return {};
        }

        // [optional]
        using scalar_type = T;

        // [required]
        static constexpr auto evaluate(scalar_type const& x) -> scalar_type const& {
            return x;
        }

        // [optional]
        static constexpr auto evaluate(scalar_type& x) -> scalar_type& {
            return x;
        }

        // [required]
        static constexpr auto outer() -> tensor_index<> {
            return {};
        }
    };

    template <concepts::tensor T>
    struct traits<std::vector<T>>
    {
        // [optional]
        using scalar_type = ttl::scalar_type<T>;

        // [required]
        static constexpr auto extents(std::vector<T> const& x)
            TTL_ARROW ( _prepend_extent<std::dynamic_extent>(ttl::extents(x[0]), x.size()) );

        // [required]
        static constexpr auto evaluate(std::vector<T> const& x, std::size_t i, std::integral auto... j)
            TTL_ARROW ( ttl::evaluate(x[i], j...) );

        // [optional]
        static constexpr auto evaluate(std::vector<T>& x, std::size_t i, std::integral auto... j)
            TTL_ARROW ( ttl::evaluate(x[i], j...) );
    };

    template <concepts::tensor T, std::size_t N>
    struct traits<T[N]>
    {
        // [optional]
        using scalar_type = ttl::scalar_type<T>;

        // [required]
        static consteval auto extents(T const (&x)[N])
            TTL_ARROW ( _prepend_extent<N>(ttl::extents(x[0])) );

        // [required]
        static constexpr auto evaluate(T const(&x)[N], std::size_t i, std::integral auto... j)
            TTL_ARROW ( ttl::evaluate(x[i], j...) );

        // [optional]
        static constexpr auto evaluate(T(&x)[N], std::size_t i, std::integral auto... j)
            TTL_ARROW ( ttl::evaluate(x[i], j...) );
    };

    template <concepts::tensor T, std::size_t N>
    struct traits<T const[N]> : traits<T[N]> {};

    template <concepts::tensor T, std::size_t N>
    struct traits<std::array<T, N>>
    {
        // [optional]
        using scalar_type = ttl::scalar_type<T>;

        // [required]
        static constexpr auto extents(std::array<T, N> const& x)
            TTL_ARROW ( _prepend_extent<N>(ttl::extents(x[0]), x.size()) );

        // [required]
        static constexpr auto evaluate(std::array<T, N> const& x, std::size_t i, std::integral auto... j)
            TTL_ARROW ( ttl::evaluate(x[i], j...) );

        // [optional]
        static constexpr auto evaluate(std::array<T, N>& x, std::size_t i, std::integral auto... j)
            TTL_ARROW ( ttl::evaluate(x[i], j...) );
    };

    template <concepts::tensor T, std::size_t N>
    struct traits<std::span<T, N>>
    {
        // [optional]
        using scalar_type = ttl::scalar_type<T>;

        // [required]
        static constexpr auto extents(std::span<T, N> const& x)
            TTL_ARROW ( _prepend_extent<N>(ttl::extents(x[0]), x.size()) );

        // [required]
        static constexpr auto evaluate(std::span<T, N> const& x, std::size_t i, std::integral auto... j)
            TTL_ARROW ( ttl::evaluate(x[i], j...) );

        // [optional]
        static constexpr auto evaluate(std::span<T, N>& x, std::size_t i, std::integral auto... j)
            TTL_ARROW ( ttl::evaluate(x[i], j...) );
    };

    template <concepts::scalar T, class Extents, class LayoutPolicy, class AccessorPolicy>
    struct traits<std::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>>
    {
        using mdspan = std::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>;

        // [optional]
        using scalar_type = ttl::scalar_type<T>;

        // [required]
        static constexpr auto extents(mdspan const& x)
            TTL_ARROW ( x.extents() );

        // [required]
        static constexpr auto evaluate(mdspan const& x, std::integral auto... i)
            TTL_ARROW ( x[i...] );

        // [optional]
        static constexpr auto evaluate(mdspan& x, std::integral auto... i)
            TTL_ARROW ( x[i...] );
    };
}
