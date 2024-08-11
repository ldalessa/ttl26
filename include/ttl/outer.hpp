#pragma once

#include <ttl/index.hpp>
#include <ttl/tensor.hpp>

namespace ttl
{
    namespace _
    {
        template <class>
        struct outer {
        };

        template <scalar T>
        struct outer<T> {
            static constexpr ttl::index_string value = {};
        };

        template <class T>
            requires(not scalar<T> and requires { std::remove_reference_t<T>::outer(); })
        struct outer<T> {
            static constexpr ttl::index_string value = std::remove_reference_t<T>::outer();
        };
    }

    template <class T>
        requires requires { _::outer<T>::value; }
    inline constexpr auto outer = _::outer<T>::value;
}
