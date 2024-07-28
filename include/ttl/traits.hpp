#pragma once

#include <concepts>
#include <cstddef>

namespace ttl
{
    namespace traits {
        template <class> struct tensor{};
        template <class T> struct tensor<T const> : tensor<T>{};
        template <class T> struct tensor<T&> : tensor<T>{};
        template <class T> struct tensor<T&&> : tensor<T>{};

        template <class> struct expression{};
        template <class T> struct expression<T const> : expression<T>{};
        template <class T> struct expression<T&> : expression<T>{};
        template <class T> struct expression<T&&> : expression<T>{};
    }
}
