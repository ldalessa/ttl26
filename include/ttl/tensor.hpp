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
        concept tensor_extents = requires (T&& t) {
            { std::remove_reference_t<T>::rank() } -> std::convertible_to<std::size_t>;
            { t.extent(0) } -> std::convertible_to<std::size_t>;
        };
    }

    /// Tensor traits:
    ///   [required] traits<T>::extents(t) -> tensor_extents
    ///   [required] traits<T const>::evaluate(t, i...)
    ///
    /// Expression traits:
    ///   [required] outer<T>::outer() -> tensor_index
    template <class T> struct traits;
    template <class T> struct traits<T const> : traits<T>{};
    template <class T> struct traits<T&>      : traits<T>{};
    template <class T> struct traits<T&&>     : traits<T>{};

    inline constexpr struct _has_extents_fn
    {
        template <class T>
        static constexpr auto operator()(T&& t)
            TTL_ARROW ( traits<T>::extents(std::forward<T>(t)) );
    } extents;

    template<class T>
    using extents_type = std::remove_reference_t<std::invoke_result_t<decltype(traits<T>::extents), T&&>>;

    template <class T>
    inline constexpr std::size_t rank = extents_type<T>::rank();

    inline constexpr struct _evaluate_fn
    {
        template <class T, std::integral... Is>
        requires (sizeof...(Is) == rank<T>)
        static constexpr auto operator()(T&& t, Is... i)
            TTL_ARROW ( traits<T>::evaluate(std::forward<T>(t), i...) );

        template <class T, std::size_t N, std::size_t... i>
        static constexpr auto operator()(T&& t, std::array<std::size_t, N> const& index, std::index_sequence<i...>)
            TTL_ARROW ( operator()(std::forward<T>(t), index[i]...) );

        template <class T, std::size_t N>
        static constexpr auto operator()(T&& t, std::array<std::size_t, N> const& index)
            TTL_ARROW ( operator()(std::forward<T>(t), index, std::make_index_sequence<N>()) );
    } evaluate;

    template <class T>
    using scalar_type = std::remove_cvref_t<std::invoke_result_t<_evaluate_fn, T&&, std::array<std::size_t, rank<T>>>>;

    template <class T>
    inline constexpr auto outer = traits<T>::outer();

    /// Promary concepts for detecting tensors and expressions.
    namespace concepts
    {
        template <class T>
        concept tensor = requires (T&& t) {
            { ttl::extents(std::forward<T>(t)) } -> tensor_extents;
            requires requires (std::array<std::size_t, ttl::rank<T>> index) {
                ttl::evaluate(std::forward<T>(t), index);
            };
        };

        template <class T, std::size_t N>
        concept tensor_of_rank = tensor<T> and rank<T> == N;

        template <class T>
        concept scalar = tensor_of_rank<T, 0zu>;

        template <class T>
        concept expression = tensor<T> and requires {
            { ttl::outer<T> } -> tensor_index;
        };
    }
}
