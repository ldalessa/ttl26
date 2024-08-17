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

#include <cassert>
#include <cstddef>
#include <functional>
#include <mdspan>
#include <type_traits>

module ttl:extents;
import :concepts;
import :tensor_traits;

template <class T>
concept has_extents_trait = requires (T&& t) {
    { ttl::tensor_traits<std::remove_cvref_t<T>>::extents(FWD(t)) } -> std_extents;
};

inline constexpr class
{
    /// Local helper variable template to deduce the extent for a range.
    template <std::ranges::range>
    struct _extent : std::integral_constant<std::size_t, std::dynamic_extent> {};

    template <c_array T>
    struct _extent<T> : std::extent<std::remove_cvref_t<T>, 0> {};

    template <std_array T>
    struct _extent<T> : std::tuple_size<std::decay_t<T>> {};

    template <std_span T>
    struct _extent<T> : std::integral_constant<std::size_t, std::decay_t<T>::extent> {};

    template <std::size_t a, class T, std::size_t... bs>
    static constexpr auto _prepend(T x, std::extents<T, bs...> const& b)
        -> std::extents<T, a, bs...>
    {
        int i = 0;
        return std::extents<T, a, bs...> { x, b.extent(((void)bs, i++))... };
    }

    template <std::ranges::range R, std::size_t... bs>
    static constexpr auto _prepend(R&& r, std::extents<std::size_t, bs...> const& b)
    {
        static constexpr std::size_t N = _extent<R>::value;
        auto const n = std::ranges::size(FWD(r));
        return _prepend<N>(n, b);
    }

  public:
    static constexpr auto operator()(integral auto) -> std::extents<std::size_t> {
        return {};
    }

    static constexpr auto operator()(floating_point auto) -> std::extents<std::size_t> {
        return {};
    }

    template <has_extents_trait T>
    static constexpr auto operator()(T&& t)
        -> ARROW( ttl::tensor_traits<std::remove_cvref_t<T>>::extents(FWD(t)) );

    template <std::ranges::range Range>
    requires (not has_extents_trait<Range>)
    constexpr auto operator()(this auto self, Range&& a)
        -> ARROW( _prepend(FWD(a), self(*std::ranges::begin(a))) );

    template <class T>
    requires (not has_extents_trait<T> and not std::ranges::range<T>)
    static constexpr auto operator()(T&& t)
        -> ARROW( FWD(t).extents() );
} extents;

template <class T>
concept has_extents = requires (T&& t) {
    { extents(FWD(t)) } -> std_extents;
};

template <class T, std::size_t... as, class U, std::size_t... bs>
inline constexpr auto concat_extents(std::extents<T, as...> const& a, std::extents<U, bs...> const& b)
    -> std::extents<std::common_type_t<T, U>, as..., bs...>
{
    using std::size_t;
    using std::extents;
    using V = std::common_type_t<T, U>;

    int i = 0;
    int j = 0;
    return extents<V, as..., bs...> { a.extent(((void)as, i++))..., b.extent(((void)bs, j++))... };
}

template <std::size_t... i, class T, std::size_t... ts>
inline constexpr auto select_extents(std::index_sequence<i...> const&, std::extents<T, ts...> const& t)
{
    using std::array;
    using std::extents;
    using std::size_t;

    static constexpr size_t Rank = sizeof...(ts);
    static constexpr array<size_t, Rank> es { ts... };
    return extents<size_t, es[i]...> { t.extent(i)... };
}

template <class T, std::size_t... as, std::size_t... bs>
inline constexpr bool compatible_extents(std::extents<T, as...> const& a, std::extents<T, bs...> const& b)
{
    using std::size_t;
    using std::dynamic_extent;
    using std::index_sequence;
    using std::make_index_sequence;

    static constexpr size_t N = sizeof...(as);
    static constexpr size_t M = sizeof...(bs);
    static_assert(N == M);
    static_assert(
            ((as == bs or as == dynamic_extent or bs == dynamic_extent) && ...),
            "Extents are incompatible.");

    return [&]<size_t... i>(index_sequence<i...>) {
        return ((a.extent(i) == b.extent(i)) && ...);
    }(make_index_sequence<N>());
}

template <class T, std::size_t... as, std::size_t... bs>
inline constexpr auto merge_extents(std::extents<T, as...> const& a, std::extents<T, bs...> const& b)
    -> std::extents<T, std::min(as, bs)...>
{
    assert(compatible_extents(a, b));
    return std::extents<T, std::min(as, bs)...> { a };
}

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

static_assert(has_extents<std::array<int[3], 3>>);
static_assert(has_extents<std::array<int[3][3], 3>&>);
static_assert(has_extents<std::array<int[3][3][3], 3>&>);

static_assert(has_extents<std::vector<int>>);
static_assert(has_extents<std::vector<int>&>);
static_assert(has_extents<std::vector<int>&&>);

static_assert(has_extents<std::vector<int> const>);
static_assert(has_extents<std::vector<int> const&>);
static_assert(has_extents<std::vector<int> const&&>);

static_assert(has_extents<std::vector<std::array<int, 3>>>);
static_assert(has_extents<std::vector<std::array<int[3], 3>>>);

static_assert(has_extents<std::mdspan<int, std::extents<std::size_t>>>);
static_assert(has_extents<std::mdspan<int, std::extents<std::size_t, 1>>>);
static_assert(has_extents<std::mdspan<int, std::extents<std::size_t, 2, 3>>>);

#undef DNDEBUG

static constexpr bool check_extents()
{
    static constexpr std::extents<std::size_t> scalar;
    static constexpr std::extents<std::size_t, 3> vector_3;
    static constexpr std::extents<std::size_t, 3, 3> vector_3_3;

    static constexpr std::extents<std::size_t, std::dynamic_extent> vector_n3(3);
    static constexpr std::extents<std::size_t, std::dynamic_extent, 3> vector_n3_3(3, 3);
    static constexpr std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent> vector_n3_n3(3, 3);

    assert(extents(static_cast<int>(1)) == scalar);
    assert(extents(static_cast<int const>(1)) == scalar);
    assert(extents(static_cast<int const&&>(1)) == scalar);

    int a = 1;
    assert(extents(static_cast<int&>(a)) == scalar);
    assert(extents(static_cast<int&&>(a)) == scalar);
    assert(extents(static_cast<int const&>(a)) == scalar);
    assert(extents(static_cast<int const&&>(a)) == scalar);

    int b[3]{};
    assert(extents(std::span(b)) == vector_3);
    assert(extents(std::span(std::as_const(b))) == vector_3);

    assert(extents(std::mdspan(b, 3)) == vector_n3);
    assert(extents(std::mdspan(std::as_const(b), 3)) == vector_n3);

    assert(extents(static_cast<int(&)[3]>(b)) == vector_3);
    assert(extents(static_cast<int(&&)[3]>(b)) == vector_3);
    assert(extents(static_cast<int const(&)[3]>(b)) == vector_3);
    assert(extents(static_cast<int const(&&)[3]>(b)) == vector_3);

    int c[3][3]{};
    assert(extents(std::span(c)) == vector_3_3);
    assert(extents(std::span(std::as_const(c))) == vector_3_3);

    assert(extents(std::mdspan(b, 3, 3)) == vector_n3_n3);
    assert(extents(std::mdspan(std::as_const(b), 3, 3)) == vector_n3_n3);

    assert(extents(static_cast<int(&)[3][3]>(c)) == vector_3_3);
    assert(extents(static_cast<int(&&)[3][3]>(c)) == vector_3_3);
    assert(extents(static_cast<int const(&)[3][3]>(c)) == vector_3_3);
    assert(extents(static_cast<int const(&&)[3][3]>(c)) == vector_3_3);

    std::array<int, 3> d{};
    assert(extents(std::span(d)) == vector_3);
    assert(extents(std::span(std::as_const(d))) == vector_3);

    assert(extents(d) == vector_3);
    assert(extents(std::as_const(d)) == vector_3);
    assert(extents(std::move(d)) == vector_3);

    std::array<int const, 3> e{};
    assert(extents(std::span(e)) == vector_3);
    assert(extents(std::span(std::as_const(e))) == vector_3);

    assert(extents(e) == vector_3);
    assert(extents(std::as_const(e)) == vector_3);
    assert(extents(std::move(e)) == vector_3);

    std::array<std::array<int, 3>, 3> f{};
    assert(extents(std::span(f)) == vector_3_3);
    assert(extents(std::span(std::as_const(f))) == vector_3_3);

    assert(extents(f) == vector_3_3);
    assert(extents(std::as_const(f)) == vector_3_3);
    assert(extents(std::move(f)) == vector_3_3);

    std::vector<int> g(3);
    assert(extents(std::span(g)) == vector_n3);
    assert(extents(std::span(std::as_const(g))) == vector_n3);

    assert(extents(g) == vector_n3);
    assert(extents(std::as_const(g)) == vector_n3);
    assert(extents(std::move(g)) == vector_n3);

    std::vector<std::array<int, 3>> h(3);
    assert(extents(std::span(h)) == vector_n3_3);
    assert(extents(std::span(std::as_const(h))) == vector_n3_3);

    assert(extents(h) == vector_n3_3);
    assert(extents(std::as_const(h)) == vector_n3_3);
    assert(extents(std::move(h)) == vector_n3_3);

    return true;
}

static_assert(check_extents());
