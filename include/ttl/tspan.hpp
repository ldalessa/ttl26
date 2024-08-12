#pragma once

#include <ttl/bind.hpp>
#include <ttl/index.hpp>
#include <ttl/tensor.hpp>
#include <ttl/tensor_traits.hpp>
#include <ttl/tree/assign.hpp>

#include <concepts>
#include <cstddef>
#include <iterator>
#include <mdspan>
#include <ranges>
#include <span>
#include <utility>

namespace ttl
{
    /// A little concept to constrain/SFINAE on std::extents.
    template <class T>
    concept std_extents = requires(T& t) {
        []<class U, std::size_t... i>(std::extents<U, i...>) {}(t);
    };

    template <
        class T,
        class Extents,
        class LayoutPolicy = std::layout_right,
        class AccessorPolicy = std::default_accessor<T>>
    struct tspan : public std::mdspan<T, Extents, LayoutPolicy, AccessorPolicy> {
        static_assert(scalar<T>);
        static_assert(std_extents<Extents>);

        /// Use all of the mdspan constructors.
        using tspan::mdspan::mdspan;

        /// Construct a tspan from an mdspan.
        constexpr tspan(std::mdspan<T, Extents, LayoutPolicy, AccessorPolicy> mdspan)
            : tspan::mdspan(std::move(mdspan))
        {
        }

        /// Construct a tspan for a congtiguous range.
        /// @{
        template <class R>
            requires std::ranges::contiguous_range<R> and std::ranges::sized_range<R>
        constexpr tspan(R&& r)
            : tspan::mdspan(std::ranges::data(r), std::ranges::size(r))
        {
        }

        constexpr tspan(std::ranges::contiguous_range auto&& r, Extents extents)
            : tspan::mdspan(std::ranges::data(r), std::move(extents))
        {
        }

        constexpr tspan(std::ranges::contiguous_range auto&& r, std::size_t i, std::integral auto... j)
            : tspan::mdspan(std::ranges::data(r), Extents(i, j...))
        {
        }
        /// @}

        /// Construct a tspan for a congtiguous iterator.
        /// @{
        constexpr tspan(std::contiguous_iterator auto it, Extents extents)
            : tspan::mdspan(std::to_address(it), std::move(extents))
        {
        }

        constexpr tspan(std::contiguous_iterator auto it, std::size_t i, std::integral auto... j)
            : tspan::mdspan(std::to_address(it), Extents(i, j...))
        {
        }
        /// @}

        /// Assign from a tensor.
        template <class A>
        constexpr auto operator=(this A&& a, tensor auto&& b) -> decltype(a) {
            ttl::tree::assign(__fwd(a), __fwd(b));
            return a;
        }

        /// Tensor indexing.
        ///
        /// This creates a bind node for an mdspan... tspans are never
        /// themselves bound. This allows us to write custom evaluation code for
        /// mdspan and have it work properly for things bound via the tspan.
        ///
        /// Copying the mdspan is going to be cheap since it's non- owning.
        constexpr auto operator()(this auto&& self, is_index auto... i)
            -> decltype(ttl::bind(typename tspan::mdspan(__fwd(self)), ttl::index(i)...))
        {
            static_assert(sizeof...(i) == Extents::rank());
            return ttl::bind(typename tspan::mdspan(__fwd(self)), ttl::index(i)...);
        }
    };

    /// Infer the scalar type and extents for a c-array.
    template <class T, std::size_t N>
    tspan(T (&)[N]) -> tspan<T, std::extents<std::size_t, N>>;

    /// Infer the scalar type and static extents for an array.
    /// @{
    template <class T, std::size_t N>
    tspan(std::array<T, N>&) -> tspan<T, std::extents<std::size_t, N>>;

    template <class T, std::size_t N>
    tspan(std::array<T, N> const&) -> tspan<std::add_const_t<T>, std::extents<std::size_t, N>>;
    /// @}

    /// Infer the scalar type and static extents for a span.
    template <class T, std::size_t N>
    tspan(std::span<T, N>) -> tspan<T, std::extents<std::size_t, N>>;

    /// Infer the scalar type and dynamic extents for a contiguous range.
    template <class R>
        requires std::ranges::contiguous_range<R> and std::ranges::sized_range<R>
    tspan(R&&)
        -> tspan<
            std::remove_reference_t<std::ranges::range_reference_t<R>>,
            std::extents<std::size_t, std::dynamic_extent>>;

    /// Infer T as the range value type.
    template <std::ranges::contiguous_range R, std_extents Extents>
    tspan(R&&, Extents const&)
        -> tspan<
            std::remove_reference_t<std::ranges::range_reference_t<R>>,
            Extents>;

    /// Infer T as the range value type, and the dynamic extents.
    template <std::ranges::contiguous_range R>
    tspan(R&&, std::integral auto, std::integral auto... i)
        -> tspan<
            std::remove_reference_t<std::ranges::range_reference_t<R>>,
            std::extents<std::size_t, std::dynamic_extent, ((void)i, std::dynamic_extent)...>>;

    /// Infer T as the iterator value type.
    template <std::contiguous_iterator It, std_extents Extents>
    tspan(It const&, Extents const&)
        -> tspan<
            std::remove_reference_t<std::iter_reference_t<It>>,
            Extents>;

    /// Infer T as the iterator value type, and the dynamic extents.
    template <std::contiguous_iterator It, std::integral... I>
    tspan(It const&, std::integral auto... i)
        -> tspan<
            std::remove_reference_t<std::iter_reference_t<It>>,
            std::extents<std::size_t, ((void)i, std::dynamic_extent)...>>;
}
