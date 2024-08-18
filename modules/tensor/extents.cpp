/// Functionality relating to extents in TTL.
///
/// There are two major components implemented in this partition.
///
/// 1. The extents() function.
///
///   The extents function is one of the two main interfaces that defines the
///   tensor concept. The extent call will return the extents for a tensor as a
///   std::extent. User types can customize this behavior by implementing
///   std::tensor_traits<T>::extents(t), or implement t.extents() ->
///   std_extents.
///
/// 2. Utilities for dealing with std::extent.
///
///    Extents are instances of std::extent, and we need to merge and filter
///    them up and down the expression trees, so this file has functions for
///    contatenation and selection of extents.
///
module;

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <mdspan>
#include <ranges>
#include <type_traits>
#include <vector>

module ttl:extents;
import :concepts;
import :tensor_traits;

namespace stdr = std::ranges;

namespace ttl
{
    /// Implement the extents() overload set as a function object.
    inline constexpr class _extents_fn
    {
        /// Template to deduce the extent for a range.
        ///
        /// This normalizes across the ways to get the three different
        /// statically-sized ranges, and uses std::dynamic_extent for variable sized
        /// ranges.
        ///
        /// * C arrays: std::extent<>
        /// * std::span: (s).extent
        /// * std::array: std::tuple_size_v
        ///
        /// @{
        template <stdr::range>
        static constexpr auto _extent = std::dynamic_extent;

        template <concepts::c_array T>
        static constexpr auto _extent<T> = std::extent<std::remove_cvref_t<T>, 0>::value;

        template <concepts::array T>
        static constexpr auto _extent<T> = std::tuple_size<std::decay_t<T>>::value;

        template <concepts::span T>
        static constexpr auto _extent<T> = std::decay_t<T>::extent;
        /// @}

        /// Prepend an extent to an existing extent.
        template <std::size_t a, class T, std::size_t... bs>
        static constexpr auto _prepend(T x, std::extents<T, bs...> const& b)
            -> std::extents<T, a, bs...>
        {
            int i = 0;
            return std::extents<T, a, bs...> { x, b.extent(((void)bs, i++))... };
        }

        /// Prepend the extent of a range to an existing extent.
        template <stdr::range R, class T, std::size_t... bs>
        static constexpr auto _prepend(R&& r, std::extents<T, bs...> const& b)
            -> ARROW( _prepend<_extent<R>>(stdr::size(r), b) );

      public:
        template <class T>
            requires (concepts::integral<T> or concepts::floating_point<T>)
        static constexpr auto operator()(T&&) -> std::extents<std::size_t> {
            return {};
        }

        /// Overload for types with defined traits.
        template <concepts::has_extents_trait T>
        static constexpr auto operator()(T&& t)
            -> ARROW( ttl::tensor_traits<std::remove_cvref_t<T>>::extents(FWD(t)) );

        /// Overload for ranges without defined traits.
        template <stdr::range Range>
        requires (not concepts::has_extents_trait<Range>)
        constexpr auto operator()(this auto self, Range&& a)
            -> ARROW( _prepend(FWD(a), self(*stdr::begin(a))) );

        /// Overload for types with .extents() member functions.
        template <class T>
        requires (not concepts::has_extents_trait<T> and not stdr::range<T>)
        static constexpr auto operator()(T&& t)
            -> ARROW( FWD(t).extents() );
    } extents;

    namespace concepts
    {
        template <class T>
        concept has_extents = requires (T&& t) {
            { ttl::extents(FWD(t)) } -> extents;
        };
    }

    /// Utilities for dealing with std::extents.
    ///@{

    /// Concatenate two sets of extents.
    template <class T, std::size_t... as, class U, std::size_t... bs>
    inline constexpr auto concat_extents(std::extents<T, as...> const& a, std::extents<U, bs...> const& b)
        -> std::extents<std::common_type_t<T, U>, as..., bs...>
    {
        int i = 0;
        int j = 0;
        return std::extents<std::common_type_t<T, U>, as..., bs...> {
            a.extent(((void)as, i++))...,
            b.extent(((void)bs, j++))...
        };
    }

    /// Select a subset of extents.
    template <std::size_t... i, class T, std::size_t... ts>
    inline constexpr auto select_extents(std::index_sequence<i...>, std::extents<T, ts...> const& t)
    {
        static constexpr std::size_t Rank = sizeof...(ts);
        static constexpr std::array<size_t, Rank> es { ts... };
        return std::extents<std::size_t, es[i]...> { t.extent(i)... };
    }

    /// Check to see if two extents are compatible.
    ///
    /// Extents are compatible when their extent value type is the same and
    /// their corresponding extent matches. The match may either be static or
    /// dynamic.
    ///
    /// ```c++
    /// std::extents<int, 1, std::dynamic_extent>(3) a;
    /// std::extents<int, std::dynamic_extent, 3>(1) b;
    /// assert(compatible_extents(a, b) == true);
    /// ```
    template <class T, std::size_t... as, std::size_t... bs>
    inline constexpr bool compatible_extents(std::extents<T, as...> const& a, std::extents<T, bs...> const& b)
    {
        /// 1. Extents should be the same length.
        static constexpr size_t N = sizeof...(as);
        static constexpr size_t M = sizeof...(bs);
        static_assert(N == M);

        /// 2. Each static extent needs to be the same, or std::dynamic_extent.
        static_assert(((as == bs or std::max(as, bs) == std::dynamic_extent) && ...),
                      "Extents are incompatible.");

        /// 3. Each dynamic extent needs to match.
        return [&]<std::size_t... i>(std::index_sequence<i...>) {
            return ((a.extent(i) == b.extent(i)) && ...);
        }(std::make_index_sequence<N>());
    }

    /// Merge two extents.
    ///
    /// This takes two compatible extents and "merges" them, which means that,
    /// for each extent, if at least one of the extents is static, then the
    /// merged extent is static.
    ///
    /// ```c++
    /// std::extents<int, 1, std::dynamic_extent>(3) a;
    /// std::extents<int, std::dynamic_extent, 3>(1) b;
    /// assert(compatible_extents(a, b));
    /// auto c = merge_extents(a, b);
    /// static_assert(std::same_as<decltype(x), std::extents<int, 1, 3>);
    /// ```
    ///
    /// @precondition `compatible_extents(a, b)`
    template <class T, std::size_t... as, std::size_t... bs>
    inline constexpr auto merge_extents(std::extents<T, as...> const& a, std::extents<T, bs...> const& b)
        -> std::extents<T, std::min(as, bs)...>
    {
        assert(compatible_extents(a, b));
        return std::extents<T, std::min(as, bs)...> { a };
    }
}

using namespace ttl;
using namespace ttl::concepts;

#undef DNDEBUG

static_assert(has_extents<int>);
static_assert(has_extents<int&>);
static_assert(has_extents<int&&>);

static_assert(has_extents<int const>);
static_assert(has_extents<int const&>);
static_assert(has_extents<int const&&>);

static_assert(has_extents<int[3]>);
static_assert(has_extents<int(&)[3]>);
static_assert(has_extents<int(&&)[3]>);

static_assert(has_extents<int const[3]>);
static_assert(has_extents<int const(&)[3]>);
static_assert(has_extents<int const(&&)[3]>);

static_assert(has_extents<int[3][3]>);
static_assert(has_extents<int(&)[3][3]>);
static_assert(has_extents<int(&&)[3][3]>);

static_assert(has_extents<int const[3]>);
static_assert(has_extents<int const(&)[3]>);
static_assert(has_extents<int const(&&)[3]>);

static_assert(has_extents<float>);

static_assert(has_extents<std::span<int, 3>>);
static_assert(has_extents<std::span<int const, 3>>);
static_assert(has_extents<std::span<int[3], 3>>);
static_assert(has_extents<std::span<int[3][3], 3>>);

static_assert(has_extents<std::array<int, 3>>);
static_assert(has_extents<std::array<int, 3>&>);
static_assert(has_extents<std::array<int, 3>&&>);

static_assert(has_extents<std::array<int, 3> const>);
static_assert(has_extents<std::array<int, 3> const&>);
static_assert(has_extents<std::array<int, 3> const&&>);

static_assert(has_extents<std::array<int const, 3>>);
static_assert(has_extents<std::array<int const, 3>&>);
static_assert(has_extents<std::array<int const, 3>&&>);

static_assert(has_extents<std::array<int const, 3> const>);
static_assert(has_extents<std::array<int const, 3> const&>);
static_assert(has_extents<std::array<int const, 3> const&&>);

static_assert(has_extents<std::array<int[3], 3>>);
static_assert(has_extents<std::array<int[3][3], 3>&>);
static_assert(has_extents<std::array<int[3][3][3], 3>&>);

static_assert(has_extents<std::array<int const[3], 3>>);
static_assert(has_extents<std::array<int const[3][3], 3>&>);
static_assert(has_extents<std::array<int const[3][3][3], 3>&>);

static_assert(has_extents<std::vector<int>>);
static_assert(has_extents<std::vector<int>&>);
static_assert(has_extents<std::vector<int>&&>);

static_assert(has_extents<std::vector<int> const>);
static_assert(has_extents<std::vector<int> const&>);
static_assert(has_extents<std::vector<int> const&&>);

static_assert(has_extents<std::vector<std::vector<int>>>);
static_assert(has_extents<std::vector<std::vector<int>>&>);
static_assert(has_extents<std::vector<std::vector<int>>&&>);

static_assert(has_extents<std::vector<std::vector<int>> const>);
static_assert(has_extents<std::vector<std::vector<int>> const&>);
static_assert(has_extents<std::vector<std::vector<int>> const&&>);

static_assert(has_extents<std::vector<std::array<int, 3>>>);
static_assert(has_extents<std::vector<std::array<int[3], 3>>>);

static_assert(has_extents<std::mdspan<int, std::extents<std::size_t>>>);
static_assert(has_extents<std::mdspan<int, std::extents<std::size_t, 1>>>);
static_assert(has_extents<std::mdspan<int, std::extents<std::size_t, 2, 3>>>);

static constexpr bool check_extents()
{
    static constexpr std::extents<std::size_t> scalar;
    static constexpr std::extents<std::size_t, 3> vector_3;
    static constexpr std::extents<std::size_t, 3, 3> vector_3_3;

    static constexpr std::extents<std::size_t, std::dynamic_extent> vector_n3(3);
    static constexpr std::extents<std::size_t, std::dynamic_extent, 3> vector_n3_3(3, 3);
    static constexpr std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent> vector_n3_n3(3, 3);

    assert(ttl::extents(static_cast<int>(1)) == scalar);
    assert(ttl::extents(static_cast<int const>(1)) == scalar);
    assert(ttl::extents(static_cast<int const&&>(1)) == scalar);

    int a = 1;
    assert(ttl::extents(static_cast<int&>(a)) == scalar);
    assert(ttl::extents(static_cast<int&&>(a)) == scalar);
    assert(ttl::extents(static_cast<int const&>(a)) == scalar);
    assert(ttl::extents(static_cast<int const&&>(a)) == scalar);

    int b[3]{};
    assert(ttl::extents(std::span(b)) == vector_3);
    assert(ttl::extents(std::span(std::as_const(b))) == vector_3);

    assert(ttl::extents(std::mdspan(b, 3)) == vector_n3);
    assert(ttl::extents(std::mdspan(std::as_const(b), 3)) == vector_n3);

    assert(ttl::extents(static_cast<int(&)[3]>(b)) == vector_3);
    assert(ttl::extents(static_cast<int(&&)[3]>(b)) == vector_3);
    assert(ttl::extents(static_cast<int const(&)[3]>(b)) == vector_3);
    assert(ttl::extents(static_cast<int const(&&)[3]>(b)) == vector_3);

    int c[3][3]{};
    assert(ttl::extents(std::span(c)) == vector_3_3);
    assert(ttl::extents(std::span(std::as_const(c))) == vector_3_3);

    assert(ttl::extents(std::mdspan(b, 3, 3)) == vector_n3_n3);
    assert(ttl::extents(std::mdspan(std::as_const(b), 3, 3)) == vector_n3_n3);

    assert(ttl::extents(static_cast<int(&)[3][3]>(c)) == vector_3_3);
    assert(ttl::extents(static_cast<int(&&)[3][3]>(c)) == vector_3_3);
    assert(ttl::extents(static_cast<int const(&)[3][3]>(c)) == vector_3_3);
    assert(ttl::extents(static_cast<int const(&&)[3][3]>(c)) == vector_3_3);

    std::array<int, 3> d{};
    assert(ttl::extents(std::span(d)) == vector_3);
    assert(ttl::extents(std::span(std::as_const(d))) == vector_3);

    assert(ttl::extents(d) == vector_3);
    assert(ttl::extents(std::as_const(d)) == vector_3);
    assert(ttl::extents(std::move(d)) == vector_3);

    std::array<int const, 3> e{};
    assert(ttl::extents(std::span(e)) == vector_3);
    assert(ttl::extents(std::span(std::as_const(e))) == vector_3);

    assert(ttl::extents(e) == vector_3);
    assert(ttl::extents(std::as_const(e)) == vector_3);
    assert(ttl::extents(std::move(e)) == vector_3);

    std::array<std::array<int, 3>, 3> f{};
    assert(ttl::extents(std::span(f)) == vector_3_3);
    assert(ttl::extents(std::span(std::as_const(f))) == vector_3_3);

    assert(ttl::extents(f) == vector_3_3);
    assert(ttl::extents(std::as_const(f)) == vector_3_3);
    assert(ttl::extents(std::move(f)) == vector_3_3);

    std::vector<int> g(3);
    assert(ttl::extents(std::span(g)) == vector_n3);
    assert(ttl::extents(std::span(std::as_const(g))) == vector_n3);

    assert(ttl::extents(g) == vector_n3);
    assert(ttl::extents(std::as_const(g)) == vector_n3);
    assert(ttl::extents(std::move(g)) == vector_n3);

    std::vector<std::array<int, 3>> h(3);
    assert(ttl::extents(std::span(h)) == vector_n3_3);
    assert(ttl::extents(std::span(std::as_const(h))) == vector_n3_3);

    assert(ttl::extents(h) == vector_n3_3);
    assert(ttl::extents(std::as_const(h)) == vector_n3_3);
    assert(ttl::extents(std::move(h)) == vector_n3_3);

    return true;
}

static_assert(check_extents());
