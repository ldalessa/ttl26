#pragma once

#include <ttl/sequence.hpp>
#include <ttl/tensor_index.hpp>

namespace ttl::tree
{
    template <tensor_index self, tensor_index b>
    inline constexpr sequence _extents_map = []<std::size_t... i>(sequence<i...>) {
        static constexpr auto map = index_of<b.rank()>(self, b);
        return sequence<map[i]...>();
    }(seqn<b.rank()>);
}
