#pragma once

#include <concepts>
#include <cstddef>

namespace ttl::concepts
{
    template <class T>
    concept size_t = std::convertible_to<T, std::size_t>;

    template <class T>
    concept tensor_index = requires {
        typename std::remove_reference_t<T>::_tensor_index_tag_t;
    };

    template <class T>
    concept tensor_extents = requires (T&& t) {
        { std::remove_reference_t<T>::rank() } -> size_t;
        { t.extent(0) } -> size_t;
    };
}
