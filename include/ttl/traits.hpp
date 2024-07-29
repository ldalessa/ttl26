/// This file defines the core tensor and expression concepts, as well as
/// functions and metafunctions to interact with them.
///
/// Tensor metafunctions:
///   ttl::rank<T>         is the rank of the tensor
///   ttl::extents_type<T> is the static extents type for a tensor
///   ttl::scalar_type<T>  is the underlying scalar type for a tensor
///
/// Tensor functions:
///   ttl::extents(t)      is the dynamic extents type for a tensor
///   ttl::evaluate(t, i...) evaluates the tensor for a specific index
///
/// Expression metafunctions:
///   ttl::outer<T>        is the outer index of the expression

#pragma once

#include <ttl/ARROW.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace ttl
{
    namespace concepts
    {
        template <class T>
        concept tensor_index = requires {
            typename std::remove_reference_t<T>::_tensor_index_tag_t;
        };

        template <class T>
        concept extents = requires (T&& t) {
            { std::remove_reference_t<T>::rank() } -> std::convertible_to<std::size_t>;
            { t.extent(0) } -> std::convertible_to<std::size_t>;
        };
    }

    /// Tensor traits:
    ///   [required] traits<T>::extents(t) -> std::extents
    ///   [required] traits<T const>::evaluate(t, i...)
    ///   [optional] traits<T>::extents_type
    ///   [optional] traits<T>::evaluate(t, i...)
    ///   [optional] traits<T>::scalar_type
    ///
    /// Expression traits:
    ///   [required] outer<T>::outer() -> tensor_index
    template <class T> struct traits;
    template <class T> struct traits<T const> : traits<T>{};
    template <class T> struct traits<T&>      : traits<T>{};
    template <class T> struct traits<T&&>     : traits<T>{};

    /// Helper concepts to figure out which traits and optional traits have been
    /// provided.
    namespace concepts
    {
        template <class T>
        concept has_extents = requires (T&& t) {
            { auto(traits<T>::extents(std::forward<T>(t))) } -> extents;
            // traits<T>::extents(std::forward<T>(t));
        };

        template <class T>
        concept has_extents_type = has_extents<T> and requires {
            typename traits<T>::extents_type;
        };

        template <class T, std::size_t N>
        concept has_evaluate = requires (T&& t) {
            []<std::size_t... i>(T&& t, std::index_sequence<i...>) TTL_ARROW (
                    traits<T>::evaluate(std::forward<T>(t), i...)
            )(std::forward<T>(t), std::make_index_sequence<N>());
        };

        template <class T, std::size_t N>
        concept has_evaluate_type = has_evaluate<T, N> and requires {
            typename traits<T>::evaluate_type;
        };

        template <class T>
        concept has_outer = requires {
            // { traits<T>::outer() } -> tensor_index;
            traits<T>::outer();
        };
    }

    /// A metafunction to get the extents_type trait.
    /// @{
    template <concepts::has_extents T>
    struct _extents_type
    {
        using type = std::remove_reference_t<std::invoke_result_t<decltype(traits<T>::extents), T&&>>;
    };

    template <concepts::has_extents_type T>
    struct _extents_type<T>
    {
        using type = traits<T>::extents_type;
    };

    template<concepts::has_extents T>
    using extents_type = _extents_type<T>::type;
    /// @}

    /// Implement the ttl::rank<T> variable template.
    /// @{
    template <concepts::has_extents T>
    inline constexpr std::size_t rank = extents_type<T>::rank();
    /// @}

    /// Promary concepts for detecting tensors and expressions.
    namespace concepts
    {
        template <class T>
        concept tensor = has_extents<T> and has_evaluate<T, rank<T>>;

        template <class T, std::size_t N>
        concept tensor_of_rank = tensor<T> and rank<T> == N;

        template <class T>
        concept scalar = tensor_of_rank<T, 0>;

        template <class T>
        concept expression = tensor<T> and has_outer<T>;
    }

    /// ttl::extents passes through to traits<T>::extents(t)
    inline constexpr struct _has_extents_fn
    {
        template <concepts::has_extents T>
        static constexpr auto operator()(T&& t)
            TTL_ARROW ( traits<T>::extents(std::forward<T>(t)) );
    } extents;

    /// ttl::evaluate passes through to traits<T>::evaluate(t, i...)
    /// - also provides a Rank-based array API rather than the variadic i...
    inline constexpr struct _evaluate_fn
    {
        template <concepts::tensor T, std::integral... Is>
        requires (sizeof...(Is) == rank<T>)
        static constexpr auto operator()(T&& t, Is... i)
            TTL_ARROW ( traits<T>::evaluate(std::forward<T>(t), i...) );

        template <concepts::tensor T, std::size_t N, std::size_t... i>
        static constexpr auto operator()(T&& t, std::array<std::size_t, N> const& index, std::index_sequence<i...>)
            TTL_ARROW ( operator()(std::forward<T>(t), index[i]...) );

        template <concepts::tensor T, std::size_t N>
        static constexpr auto operator()(T&& t, std::array<std::size_t, N> const& index)
            TTL_ARROW ( operator()(std::forward<T>(t), index, std::make_index_sequence<N>()) );
    } evaluate;

    /// A not-to-be-overloaded metafunction helper.
    template <concepts::tensor T>
    requires concepts::has_evaluate<T, rank<T>>
    struct _scalar_type {
        using type = std::remove_cvref_t<std::invoke_result_t<_evaluate_fn, T&&, std::array<std::size_t, rank<T>>>>;
    };

    template <concepts::tensor T>
    requires concepts::has_evaluate_type<T, rank<T>>
    struct _scalar_type<T> {
        using type = traits<T>::scalar_type;
    };

    /// Implement the ttl::scalar_type<T> metafunction.
    template <concepts::tensor T>
    using scalar_type = _scalar_type<std::remove_cvref_t<T>>::type;

    template <concepts::expression T>
    inline constexpr auto outer = T::outer();
}
