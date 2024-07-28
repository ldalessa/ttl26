#pragma once

#include <ttl/index.hpp>
#include <ttl/outer.hpp>
#include <ttl/tensor.hpp>
#include <ttl/traits.hpp>

namespace ttl
{
    namespace concepts
    {
        template <class T>
        concept expression = tensor<T> and has_outer<T>;
    }

    template <concepts::scalar T>
    struct traits::expression<T>
    {
        static consteval auto outer() -> ttl::index<> {
            return {};
        }
    };
}
