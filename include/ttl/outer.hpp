#pragma once

#include <ttl/index.hpp>
#include <ttl/extents.hpp>

namespace ttl
{
    namespace _
    {
        template <class>
        struct outer {
        };

        template <class T>
            requires (rank<T> == 0)
        struct outer<T> {
            static constexpr index_string value = {};
        };

        template <class T>
            requires(rank<T> != 0 and requires { std::remove_reference_t<T>::outer(); })
        struct outer<T> {
            static constexpr index_string value = std::remove_reference_t<T>::outer();
        };
    }

    template <class T>
        requires requires { _::outer<T>::value; }
    inline constexpr auto outer = _::outer<T>::value;
}
