module;

#include <array>
#include <cstddef>
#include <functional>
#include <ranges>
#include <mdspan>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

module ttl:rank;
import :extents;
import :tensor_traits;

namespace stdr = std::ranges;

namespace ttl
{
    template <std::size_t N>
    using rank_t = std::integral_constant<std::size_t, N>;

    template <std::size_t N>
    inline constexpr rank_t<N> rank_v;

    template <class T>
    inline constexpr auto rank = []
    {
        using U = std::remove_cvref_t<T>;
        if constexpr (std::integral<U> or std::floating_point<U>) {
            return rank_v<0zu>;
        }
        else if constexpr (concepts::has_rank_trait<T>) {
            return tensor_traits<U>::rank;
        }
        else if constexpr (stdr::range<U>) {
            return rank_v<rank<stdr::range_value_t<U>> + 1zu>;
        }
        else if constexpr (concepts::mdspan<U>) {
            return rank_v<rank<typename U::element_type> + U::extents_type::rank()>;
        }
        else {
            using extents_type = std::invoke_result_t<_extents_fn, U>;
            return rank_v<std::decay_t<extents_type>::rank()>;
        }
    }();
}

using namespace ttl;

#undef DNDEBUG

static_assert(rank<int> == 0);
static_assert(rank<int[1]> == 1);
static_assert(rank<int[1][1]> == 2);

static_assert(rank<int&> == 0);
static_assert(rank<int(&)[1]> == 1);
static_assert(rank<int(&)[1][1]> == 2);

static_assert(rank<int&&> == 0);
static_assert(rank<int(&&)[1]> == 1);
static_assert(rank<int(&&)[1][1]> == 2);

static_assert(rank<int const> == 0);
static_assert(rank<int const[1]> == 1);
static_assert(rank<int const[1][1]> == 2);

static_assert(rank<int const&> == 0);
static_assert(rank<int const(&)[1]> == 1);
static_assert(rank<int const(&)[1][1]> == 2);

static_assert(rank<int const&&> == 0);
static_assert(rank<int const(&&)[1]> == 1);
static_assert(rank<int const(&&)[1][1]> == 2);

static_assert(rank<std::span<int, 1>> == 1);
static_assert(rank<std::span<int[1], 1>> == 2);
static_assert(rank<std::span<int[1][1], 1>> == 3);

static_assert(rank<std::span<int const, 1>> == 1);
static_assert(rank<std::span<int const[1], 1>> == 2);
static_assert(rank<std::span<int const[1][1], 1>> == 3);

static_assert(rank<std::array<int, 1>> == 1);
static_assert(rank<std::array<int[1], 1>> == 2);
static_assert(rank<std::array<int[1][1], 1>> == 3);

static_assert(rank<std::array<int const, 1>> == 1);
static_assert(rank<std::array<int const[1], 1>> == 2);
static_assert(rank<std::array<int const[1][1], 1>> == 3);

static_assert(rank<std::vector<int>> == 1);
static_assert(rank<std::vector<int[1]>> == 2);
static_assert(rank<std::vector<int[1][1]>> == 3);

static_assert(rank<std::vector<int const>> == 1);
static_assert(rank<std::vector<int const[1]>> == 2);
static_assert(rank<std::vector<int const[1][1]>> == 3);

static_assert(rank<std::mdspan<int, std::extents<int>>> == 0);
static_assert(rank<std::mdspan<int, std::extents<int, 1>>> == 1);
static_assert(rank<std::mdspan<int, std::extents<int, 1, 1>>> == 2);
static_assert(rank<std::mdspan<int, std::extents<int, 1, std::dynamic_extent, 1>>> == 3);

static_assert(rank<std::mdspan<int const, std::extents<int>>> == 0);
static_assert(rank<std::mdspan<int const, std::extents<int, 1>>> == 1);
static_assert(rank<std::mdspan<int const, std::extents<int, 1, 1>>> == 2);
static_assert(rank<std::mdspan<int const, std::extents<int, 1, std::dynamic_extent, 1>>> == 3);

static_assert(rank<std::mdspan<std::array<int, 3>, std::extents<int>>> == 1);
static_assert(rank<std::mdspan<std::array<int, 3>, std::extents<int, 1>>> == 2);
static_assert(rank<std::mdspan<std::array<int, 3>, std::extents<int, 1, 1>>> == 3);
static_assert(rank<std::mdspan<std::array<int, 3>, std::extents<int, 1, std::dynamic_extent, 1>>> == 4);
