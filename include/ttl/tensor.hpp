#pragma once

#include <ttl/extents.hpp>
#include <ttl/evaluate.hpp>

#include <cstddef>
#include <utility>

namespace ttl
{
    template <class T>
    concept tensor = requires (T&& t) {
        ttl::extents(__fwd(t));
        requires []<std::size_t... i>(std::index_sequence<i...>) {
            return requires { ttl::evaluate(__fwd(t), ((void)i, 0)...); };
        }(std::make_index_sequence<rank<T>>());
    };

    template <class T, std::size_t N>
    concept tensor_of_rank = tensor<T> and rank<T> == N;

    template <class T>
    concept scalar = tensor_of_rank<T, 0zu>;
}
