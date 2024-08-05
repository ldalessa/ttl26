#pragma once

#include <ttl/ARROW.hpp>
#include <ttl/concepts.hpp>
#include <ttl/tensor_extents.hpp>
#include <ttl/tensor_index.hpp>
#include <ttl/tensor.hpp>
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
        static constexpr auto tensor_extents(T const&) noexcept -> std::extents<std::size_t> {
            return {};
        }

        static constexpr auto evaluate(T const& x) noexcept -> T const& {
            return x;
        }

        static constexpr auto evaluate(T& x) noexcept -> T& {
            return x;
        }

        static constexpr auto outer() noexcept -> tensor_index<> {
            return {};
        }
    };

    /// Vector traits
    template <concepts::tensor T>
    struct traits<std::vector<T>>
    {
        static constexpr auto tensor_extents(std::vector<T> const& x)
            TTL_ARROW ( prepend_extent<std::dynamic_extent>(ttl::tensor_extents(x[0]), x.size()) );

        static constexpr auto evaluate(std::vector<T> const& x, std::size_t i, concepts::size_t auto... j)
            TTL_ARROW ( ttl::evaluate(x[i], j...) );

        static constexpr auto evaluate(std::vector<T>& x, std::size_t i, concepts::size_t auto... j)
            TTL_ARROW ( ttl::evaluate(x[i], j...) );
    };

    /// Array traits
    template <concepts::tensor T, std::size_t N>
    struct traits<T[N]>
    {
        static constexpr auto tensor_extents(T const (&x)[N])
            TTL_ARROW ( prepend_extent<N>(ttl::tensor_extents(x[0])) );

        static constexpr auto evaluate(T const(&x)[N], std::size_t i, concepts::size_t auto... j)
            TTL_ARROW ( ttl::evaluate(x[i], j...) );

        static constexpr auto evaluate(T(&x)[N], std::size_t i, concepts::size_t auto... j)
            TTL_ARROW ( ttl::evaluate(x[i], j...) );
    };

    /// Constant array traits (C-arrays are weird so we need this)
    template <concepts::tensor T, std::size_t N>
    struct traits<T const[N]> : traits<T[N]> {};

    /// Array traits
    template <concepts::tensor T, std::size_t N>
    struct traits<std::array<T, N>>
    {
        static constexpr auto tensor_extents(std::array<T, N> const& x)
            TTL_ARROW ( prepend_extent<N>(ttl::tensor_extents(x[0]), x.size()) );

        static constexpr auto evaluate(std::array<T, N> const& x, std::size_t i, concepts::size_t auto... j)
            TTL_ARROW ( ttl::evaluate(x[i], j...) );

        static constexpr auto evaluate(std::array<T, N>& x, std::size_t i, concepts::size_t auto... j)
            TTL_ARROW ( ttl::evaluate(x[i], j...) );
    };

    /// Span traits
    template <concepts::tensor T, std::size_t N>
    struct traits<std::span<T, N>>
    {
        static constexpr auto tensor_extents(std::span<T, N> x)
            TTL_ARROW ( prepend_extent<N>(ttl::tensor_extents(x[0]), x.size()) );

        static constexpr auto evaluate(std::span<T, N> x, std::size_t i, concepts::size_t auto... j)
            TTL_ARROW ( ttl::evaluate(x[i], j...) );
    };

    /// mdspan traits
    ///
    /// @TODO Currently we only support mdspans with scalar element
    /// types. There's no fundamental reason we couldn't support a tensor
    /// element, it's just a somewhat noisy instantiation that I don't want to
    /// write yet.
    template <concepts::scalar T, class Extents, class LayoutPolicy, class AccessorPolicy>
    struct traits<std::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>>
    {
        using mdspan = std::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>;

        static constexpr auto tensor_extents(mdspan const& x)
            TTL_ARROW ( x.extents() );

        static constexpr auto evaluate(mdspan const& x, concepts::size_t auto... i)
            TTL_ARROW ( x[i...] );

        static constexpr auto evaluate(mdspan& x, concepts::size_t auto... i)
            TTL_ARROW ( x[i...] );
    };
}
