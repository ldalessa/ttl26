#pragma once

#include <ttl/ARROW.hpp>
#include <ttl/traits.hpp>
#include <concepts>
#include <cstddef>
#include <utility>

namespace ttl
{
    inline constexpr struct _evaluate_fn
    {
        /// We can't declare local concepts, but we _can_ evaluate concepts to
        /// booleans, which this traits structure does. If the tensor traits
        /// version of evaluate works for the type and the number of indices,
        /// then we want to use that, otherwise if the type has an indexing
        /// operator we use that.
        template <class T, std::integral... I>
        struct _traits {
            static constexpr bool use_tensor_trait = requires (T&& t) {
                traits::tensor<T>::evaluate(std::forward<T>(t), I()...);
            };

            static constexpr bool use_index_operator = not use_tensor_trait and requires (T&& t) {
                std::forward<T>(t)[I()...];
            };
        };

      public:
        /// This version is selected when we can call
        /// traits::tensor<T>::evaluate with the t, i... arguments.
        template <class T, std::integral... I>
        requires _traits<T, I...>::use_tensor_trait
        static constexpr auto operator()(T&& t, I... i)
            TTL_ARROW ( traits::tensor<T>::evaluate(std::forward<T>(t), i...) );

        /// This version is selected when we can can't find an evaluate trait
        /// but the type has an index operator that will work.
        template <class T, std::integral... I>
        requires _traits<T, I...>::use_index_operator
        static constexpr auto operator()(T&& t, I... i)
            TTL_ARROW ( std::forward<T>(t)[i...] );

      private:
        /// This just expands the array so that we can use the variadic indexing
        /// form of evaluate if we're given an array of indices.
        template <class T, std::size_t N, std::size_t... j>
        static constexpr auto _array_helper(T&& t, std::array<std::size_t, N> const& i, std::index_sequence<j...>)
            TTL_ARROW ( operator()(std::forward<T>(t), i[j]...) );

      public:
        /// This version is selected when we're being passed indices in an array
        /// rather than varidadically. This happens in the has_evaluate concept
        /// as well as possibly during generated tensor evaluation.
        template <class T, std::size_t N>
        static constexpr auto operator()(T&& t, std::array<std::size_t, N> const& i)
            TTL_ARROW ( _array_helper(std::forward<T>(t), i, std::make_index_sequence<N>()) );

    } evaluate{};

    namespace concepts
    {
        /// has_evaluate means that we can call evaluate(t, i), but this
        /// transitively means that the type either has a
        /// traits::tensor<T>::evaluate or an operator[] that can be called with
        /// N integers.
        template <class T, std::size_t N>
        concept has_evaluate = requires (T&& t, std::array<std::size_t, N> const& i) {
            ttl::evaluate(std::forward<T>(t), i);
        };
    }
}
