#pragma once

#include <ttl/index.hpp>
#include <ttl/traits.hpp>
#include <type_traits>

namespace ttl
{
    namespace _
    {
        template <class>
        struct outer;

        template <class T>
        requires requires { { traits::expression<T>::outer() } -> concepts::tensor_index; }
        struct outer<T>
        {
            static constexpr auto value = traits::expression<T>::outer();
        };

        template <class T>
        requires requires { { std::remove_reference_t<T>::outer() } -> concepts::tensor_index; }
        struct outer<T>
        {
            static constexpr auto value = std::remove_reference_t<T>::outer();
        };
    }

    namespace concepts
    {
        template <class T>
        concept has_outer = requires {
            { _::outer<T>::value } -> concepts::tensor_index;
        };
    }

    template <concepts::has_outer T>
    inline constexpr auto outer = _::outer<T>::value;
}
