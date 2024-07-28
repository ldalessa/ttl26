#pragma once

#include <ttl/traits.hpp>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace ttl
{
    namespace _
    {
        template <class>
        struct rank;

        template <class T>
        requires requires { traits::tensor<T>::rank(); }
        struct rank<T> {
            static constexpr std::size_t value = traits::tensor<T>::rank();
        };

        template <class T>
        requires requires { std::remove_reference_t<T>::rank(); }
        struct rank<T> {
            static constexpr std::size_t value = std::remove_reference_t<T>::rank();
        };
    }

    namespace concepts
    {
        template <class T>
        concept has_rank = requires {
            { _::rank<T>::value } -> std::convertible_to<std::size_t>;
        };
    }

    template <concepts::has_rank T>
    inline constexpr std::size_t rank = _::rank<T>::value;
}
