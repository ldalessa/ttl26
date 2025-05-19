#pragma once
#include <ttl/tensor_traits.hpp>
#include <ttl/tensor/extents.hpp>
import std;

namespace ttl
{
	namespace stdr = std::ranges;

    template <class T>
    inline constexpr std::size_t rank = []
    {
        using U = std::remove_cvref_t<T>;
        if constexpr (std::integral<U> or std::floating_point<U>) {
            return 0zu;
        }
        else if constexpr (concepts::has_rank_trait<T>) {
            return tensor_traits<U>::rank;
        }
        else if constexpr (stdr::range<U>) {
            return rank<stdr::range_value_t<U>> + 1zu;
        }
        else if constexpr (concepts::mdspan<U>) {
            return rank<typename U::element_type> + U::extents_type::rank();
        }
        else {
            using extents_type = std::invoke_result_t<_extents_fn, U>;
            return std::decay_t<extents_type>::rank();
        }
    }();
}
