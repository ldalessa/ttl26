module;

#include <array>
#include <cstddef>
#include <ranges>
#include <mdspan>
#include <span>
#include <type_traits>
#include <vector>

module ttl:rank;
import :extents;

/// The default for the rank implementation is to defer to the extents
/// implementation, which is what is customized by user code. This will always
/// work for valid tensors, but we can provide some short circuits for other
/// well-known types.
template <class T>
inline constexpr std::size_t rank_impl = decltype(auto(extents(std::declval<T>())))::rank();

template <std::integral T>
inline constexpr std::size_t rank_impl<T> = 0zu;

template <std::floating_point T>
inline constexpr std::size_t rank_impl<T> = 0zu;

template <std::ranges::range T>
inline constexpr std::size_t rank_impl<T> = rank_impl<std::ranges::range_value_t<T>> + 1zu;

template <class T, class Extents, class Layout, class Accessor>
inline constexpr std::size_t rank_impl<std::mdspan<T, Extents, Layout, Accessor>> = Extents::rank() + rank_impl<T>;

template <class T>
inline constexpr std::size_t rank = rank_impl<std::remove_cvref_t<T>>;

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
