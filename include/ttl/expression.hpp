#pragma once

#include <ttl/outer.hpp>
#include <ttl/tensor.hpp>

namespace ttl
{
    template <class T>
    concept expression = tensor<T> and requires {
        ttl::outer<T>;
    };
}
