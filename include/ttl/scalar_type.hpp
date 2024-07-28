#pragma once

#include <ttl/evaluate.hpp>
#include <ttl/tensor.hpp>
#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>

namespace ttl
{
    template <concepts::tensor... T>
    using scalar_type = std::common_type_t<std::remove_cvref_t<std::invoke_result_t<_evaluate_fn, T, std::array<std::size_t, ttl::rank<T>>>>...>;
}
