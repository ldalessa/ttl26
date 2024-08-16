module;

#include <array>
#include <cassert>
#include <cstddef>
#include <mdspan>
#include <ranges>
#include <span>
#include <type_traits>
#include <vector>

export module ttl:tspan;

export namespace ttl
{
    template <
        class T,
        class Extents,
        class LayoutPolicy = std::layout_right,
        class AccessorPolicy = std::default_accessor<T>>
    struct tspan : public std::mdspan<T, Extents, LayoutPolicy, AccessorPolicy> {
        /// Use all of the mdspan constructors.
        using tspan::mdspan::mdspan;

        /// Construct a tspan from an mdspan.
        constexpr tspan(std::mdspan<T, Extents, LayoutPolicy, AccessorPolicy> mdspan)
            : tspan::mdspan(std::move(mdspan))
        {
        }

        /// Construct a tspan for a congtiguous range.
        /// @{
        template <class R>
            requires std::ranges::contiguous_range<R> and std::ranges::sized_range<R>
        constexpr tspan(R&& r)
            : tspan::mdspan(std::ranges::data(r), std::ranges::size(r))
        {
        }

        constexpr tspan(std::ranges::contiguous_range auto&& r, Extents extents)
            : tspan::mdspan(std::ranges::data(r), std::move(extents))
        {
        }

        constexpr tspan(std::ranges::contiguous_range auto&& r, std::size_t i, std::integral auto... j)
            : tspan::mdspan(std::ranges::data(r), Extents(i, j...))
        {
        }
        /// @}

        /// Construct a tspan for a contiguous iterator.
        /// @{
        constexpr tspan(std::contiguous_iterator auto it, Extents extents)
            : tspan::mdspan(std::to_address(it), std::move(extents))
        {
        }

        constexpr tspan(std::contiguous_iterator auto it, std::size_t i, std::integral auto... j)
            : tspan::mdspan(std::to_address(it), Extents(i, j...))
        {
        }
        /// @}

        /// Assign from a tensor.
        // template <class A>
        // constexpr auto operator=(this A&& a, tensor auto&& b) -> decltype(a) {
        //     ttl::tree::assign(__fwd(a), __fwd(b));
        //     return a;
        // }

        /// Tensor indexing.
        ///
        /// This creates a bind node for an mdspan... tspans are never
        /// themselves bound. This allows us to write custom evaluation code for
        /// mdspan and have it work properly for things bound via the tspan.
        ///
        /// Copying the mdspan is going to be cheap since it's non- owning.
        // constexpr auto operator()(this auto&& self, is_index auto... i)
        //     -> decltype(ttl::bind(typename tspan::mdspan(__fwd(self)), ttl::index(i)...))
        // {
        //     static_assert(sizeof...(i) == Extents::rank());
        //     return ttl::bind(typename tspan::mdspan(__fwd(self)), ttl::index(i)...);
        // }
    };

    /// Infer the scalar type and static extents for a c-array.
    ///
    /// This covers both T and T const.
    template <class T, std::size_t N>
    tspan(T (&)[N]) -> tspan<T, std::extents<std::size_t, N>>;

    /// Infer the scalar type and static extents for an array.
    ///
    /// Have to infer `const` indepenently.
    /// @{
    template <class T, std::size_t N>
    tspan(std::array<T, N>&) -> tspan<T, std::extents<std::size_t, N>>;

    template <class T, std::size_t N>
    tspan(std::array<T, N> const&) -> tspan<std::add_const_t<T>, std::extents<std::size_t, N>>;
    /// @}

    /// Infer the scalar type and static extents for a span.
    ///
    /// This covers both T and T const.
    template <class T, std::size_t N>
    tspan(std::span<T, N>) -> tspan<T, std::extents<std::size_t, N>>;

    /// Infer the scalar type and dynamic extents for a contiguous range.
    template <class R>
        requires std::ranges::contiguous_range<R> and std::ranges::sized_range<R>
    tspan(R&&)
        -> tspan<
            std::remove_reference_t<std::ranges::range_reference_t<R>>,
            std::extents<std::size_t, std::dynamic_extent>>;

    /// Infer T as the range value type.
    template <std::ranges::contiguous_range R, class T, std::size_t... Es>
    tspan(R&&, std::extents<T, Es...>)
        -> tspan<
            std::remove_reference_t<std::ranges::range_reference_t<R>>,
            std::extents<T, Es...>>;

    /// Infer T as the range value type, and the dynamic extents.
    template <std::ranges::contiguous_range R>
    tspan(R&&, std::integral auto, std::integral auto... i)
        -> tspan<
            std::remove_reference_t<std::ranges::range_reference_t<R>>,
            std::extents<std::size_t, std::dynamic_extent, ((void)i, std::dynamic_extent)...>>;

    /// Infer T as the iterator value type.
    template <std::contiguous_iterator It, class T, std::size_t... Es>
    tspan(It const&, std::extents<T, Es...>)
        -> tspan<
            std::remove_reference_t<std::iter_reference_t<It>>,
            std::extents<T, Es...>>;

    /// Infer T as the iterator value type, and the dynamic extents.
    template <std::contiguous_iterator It, std::integral... I>
    tspan(It const&, std::integral auto... i)
        -> tspan<
            std::remove_reference_t<std::iter_reference_t<It>>,
            std::extents<std::size_t, ((void)i, std::dynamic_extent)...>>;
}

#undef DNDEBUG

static constexpr bool check_c_array()
{
    int a[16]{};

    ttl::tspan<int, std::extents<std::size_t, 16>> _(a);
    ttl::tspan<int, std::extents<std::size_t, 4, 4>> _(a);
    ttl::tspan<int, std::extents<std::size_t, 2, 2, 2, 2>> _(a);

    ttl::tspan<int, std::extents<std::size_t, 16>> _(a, 16);
    ttl::tspan<int, std::extents<std::size_t, 4, 4>> _(a, 4, 4);
    ttl::tspan<int, std::extents<std::size_t, 2, 2, 2, 2>> _(a, 2, 2, 2, 2);

    ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent>> _(a, 16);
    ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>> _(a, 4, 4);
    ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>> _(a, 2, 2, 2, 2);

    ttl::tspan _(a);

    ttl::tspan _(a, std::extents<std::size_t, 16>());
    ttl::tspan _(a, std::extents<std::size_t, 4, 4>());
    ttl::tspan _(a, std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(a, 16);
    ttl::tspan _(a, 4, 4);
    ttl::tspan _(a, 2, 2, 2, 2);

    auto b = std::span(a);

    ttl::tspan<int, std::extents<std::size_t, 16>> _(b);
    // ttl::tspan<int, std::extents<std::size_t, 4, 4>> _(b); @fixme constructor
    // ttl::tspan<int, std::extents<std::size_t, 2, 2, 2, 2>> _(b); @fixme constructor

    ttl::tspan<int, std::extents<std::size_t, 16>> _(b, 16);
    ttl::tspan<int, std::extents<std::size_t, 4, 4>> _(b, 4, 4);
    ttl::tspan<int, std::extents<std::size_t, 2, 2, 2, 2>> _(b, 2, 2, 2, 2);

    ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent>> _(b, 16);
    ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>> _(b, 4, 4);
    ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>> _(b, 2, 2, 2, 2);

    ttl::tspan _(b);

    ttl::tspan _(b, std::extents<std::size_t, 16>());
    ttl::tspan _(b, std::extents<std::size_t, 4, 4>());
    ttl::tspan _(b, std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(b, 16);
    ttl::tspan _(b, 4, 4);
    ttl::tspan _(b, 2, 2, 2, 2);

    auto c = std::span(a, 16);
    ttl::tspan _(c);

    ttl::tspan _(c, std::extents<std::size_t, 16>());
    ttl::tspan _(c, std::extents<std::size_t, 4, 4>());
    ttl::tspan _(c, std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(c, 16);
    ttl::tspan _(c, 4, 4);
    ttl::tspan _(c, 2, 2, 2, 2);

    int const d[16]{};
    ttl::tspan _(d);

    ttl::tspan _(d, std::extents<std::size_t, 16>());
    ttl::tspan _(d, std::extents<std::size_t, 4, 4>());
    ttl::tspan _(d, std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(d, 16);
    ttl::tspan _(d, 4, 4);
    ttl::tspan _(d, 2, 2, 2, 2);

    auto e = std::span(b);
    ttl::tspan _(e);

    ttl::tspan _(e, std::extents<std::size_t, 16>());
    ttl::tspan _(e, std::extents<std::size_t, 4, 4>());
    ttl::tspan _(e, std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(e, 16);
    ttl::tspan _(e, 4, 4);
    ttl::tspan _(e, 2, 2, 2, 2);

    auto f = std::span(b);
    ttl::tspan _(f);

    ttl::tspan _(f, std::extents<std::size_t, 16>());
    ttl::tspan _(f, std::extents<std::size_t, 4, 4>());
    ttl::tspan _(f, std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(f, 16);
    ttl::tspan _(f, 4, 4);
    ttl::tspan _(f, 2, 2, 2, 2);

    return true;
}

static constexpr bool check_array()
{
    std::array<int, 16> a{};

    ttl::tspan<int, std::extents<std::size_t, 16>> _(a);
    // ttl::tspan<int, std::extents<std::size_t, 4, 4>> _(a); @fixme constructor
    // ttl::tspan<int, std::extents<std::size_t, 2, 2, 2, 2>> _(a); @fixme constructor

    ttl::tspan<int, std::extents<std::size_t, 16>> _(a, 16);
    ttl::tspan<int, std::extents<std::size_t, 4, 4>> _(a, 4, 4);
    ttl::tspan<int, std::extents<std::size_t, 2, 2, 2, 2>> _(a, 2, 2, 2, 2);

    ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent>> _(a, 16);
    ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>> _(a, 4, 4);
    ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>> _(a, 2, 2, 2, 2);

    ttl::tspan _(a);

    ttl::tspan _(a, std::extents<std::size_t, 16>());
    ttl::tspan _(a, std::extents<std::size_t, 4, 4>());
    ttl::tspan _(a, std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(a, 16);
    ttl::tspan _(a, 4, 4);
    ttl::tspan _(a, 2, 2, 2, 2);

    ttl::tspan _(a.begin(), std::extents<std::size_t, 16>());
    ttl::tspan _(a.begin(), std::extents<std::size_t, 4, 4>());
    ttl::tspan _(a.begin(), std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(a.begin(), 16);
    ttl::tspan _(a.begin(), 4, 4);
    ttl::tspan _(a.begin(), 2, 2, 2, 2);

    ttl::tspan _(a.cbegin(), std::extents<std::size_t, 16>());
    ttl::tspan _(a.cbegin(), std::extents<std::size_t, 4, 4>());
    ttl::tspan _(a.cbegin(), std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(a.cbegin(), 16);
    ttl::tspan _(a.cbegin(), 4, 4);
    ttl::tspan _(a.cbegin(), 2, 2, 2, 2);

    const std::array<int, 16> b{};
    ttl::tspan _(b);

    ttl::tspan _(b, std::extents<std::size_t, 16>());
    ttl::tspan _(b, std::extents<std::size_t, 4, 4>());
    ttl::tspan _(b, std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(b, 16);
    ttl::tspan _(b, 4, 4);
    ttl::tspan _(b, 2, 2, 2, 2);

    ttl::tspan _(b.begin(), std::extents<std::size_t, 16>());
    ttl::tspan _(b.begin(), std::extents<std::size_t, 4, 4>());
    ttl::tspan _(b.begin(), std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(b.begin(), 16);
    ttl::tspan _(b.begin(), 4, 4);
    ttl::tspan _(b.begin(), 2, 2, 2, 2);

    auto c = std::span(b);
    ttl::tspan _(c);

    ttl::tspan _(c, std::extents<std::size_t, 16>());
    ttl::tspan _(c, std::extents<std::size_t, 4, 4>());
    ttl::tspan _(c, std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(c, 16);
    ttl::tspan _(c, 4, 4);
    ttl::tspan _(c, 2, 2, 2, 2);

    ttl::tspan _(c.begin(), std::extents<std::size_t, 16>());
    ttl::tspan _(c.begin(), std::extents<std::size_t, 4, 4>());
    ttl::tspan _(c.begin(), std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(c.begin(), 16);
    ttl::tspan _(c.begin(), 4, 4);
    ttl::tspan _(c.begin(), 2, 2, 2, 2);

    std::array<int const, 16> d{};
    ttl::tspan _(d);

    ttl::tspan _(d, std::extents<std::size_t, 16>());
    ttl::tspan _(d, std::extents<std::size_t, 4, 4>());
    ttl::tspan _(d, std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(d, 16);
    ttl::tspan _(d, 4, 4);
    ttl::tspan _(d, 2, 2, 2, 2);

    ttl::tspan _(d.begin(), std::extents<std::size_t, 16>());
    ttl::tspan _(d.begin(), std::extents<std::size_t, 4, 4>());
    ttl::tspan _(d.begin(), std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(d.begin(), 16);
    ttl::tspan _(d.begin(), 4, 4);
    ttl::tspan _(d.begin(), 2, 2, 2, 2);

    ttl::tspan _(d.cbegin(), std::extents<std::size_t, 16>());
    ttl::tspan _(d.cbegin(), std::extents<std::size_t, 4, 4>());
    ttl::tspan _(d.cbegin(), std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(d.cbegin(), 16);
    ttl::tspan _(d.cbegin(), 4, 4);
    ttl::tspan _(d.cbegin(), 2, 2, 2, 2);

    const std::array<int const, 16> e{};
    ttl::tspan _(e);

    ttl::tspan _(e, std::extents<std::size_t, 16>());
    ttl::tspan _(e, std::extents<std::size_t, 4, 4>());
    ttl::tspan _(e, std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(e, 16);
    ttl::tspan _(e, 4, 4);
    ttl::tspan _(e, 2, 2, 2, 2);

    ttl::tspan _(e.begin(), std::extents<std::size_t, 16>());
    ttl::tspan _(e.begin(), std::extents<std::size_t, 4, 4>());
    ttl::tspan _(e.begin(), std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(e.begin(), 16);
    ttl::tspan _(e.begin(), 4, 4);
    ttl::tspan _(e.begin(), 2, 2, 2, 2);

    return true;
}

static constexpr bool check_vector()
{
    std::vector<int> a(16);
    ttl::tspan _(a);

    ttl::tspan _(a, std::extents<std::size_t, 16>());
    ttl::tspan _(a, std::extents<std::size_t, 4, 4>());
    ttl::tspan _(a, std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(a, 16);
    ttl::tspan _(a, 4, 4);
    ttl::tspan _(a, 2, 2, 2, 2);

    ttl::tspan _(a.begin(), std::extents<std::size_t, 16>());
    ttl::tspan _(a.begin(), std::extents<std::size_t, 4, 4>());
    ttl::tspan _(a.begin(), std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(a.begin(), 16);
    ttl::tspan _(a.begin(), 4, 4);
    ttl::tspan _(a.begin(), 2, 2, 2, 2);

    ttl::tspan _(a.cbegin(), std::extents<std::size_t, 16>());
    ttl::tspan _(a.cbegin(), std::extents<std::size_t, 4, 4>());
    ttl::tspan _(a.cbegin(), std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(a.cbegin(), 16);
    ttl::tspan _(a.cbegin(), 4, 4);
    ttl::tspan _(a.cbegin(), 2, 2, 2, 2);

    const std::vector<int> b(16);
    ttl::tspan _(b);

    ttl::tspan _(b, std::extents<std::size_t, 16>());
    ttl::tspan _(b, std::extents<std::size_t, 4, 4>());
    ttl::tspan _(b, std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(b, 16);
    ttl::tspan _(b, 4, 4);
    ttl::tspan _(b, 2, 2, 2, 2);

    ttl::tspan _(b.begin(), std::extents<std::size_t, 16>());
    ttl::tspan _(b.begin(), std::extents<std::size_t, 4, 4>());
    ttl::tspan _(b.begin(), std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(b.begin(), 16);
    ttl::tspan _(b.begin(), 4, 4);
    ttl::tspan _(b.begin(), 2, 2, 2, 2);

    return true;
}

static constexpr bool check_new()
{
    int *a = new int[16]{};

    ttl::tspan<int, std::extents<std::size_t, 16>> _(a);
    ttl::tspan<int, std::extents<std::size_t, 4, 4>> _(a);
    ttl::tspan<int, std::extents<std::size_t, 2, 2, 2, 2>> _(a);

    ttl::tspan<int, std::extents<std::size_t, 16>> _(a, 16);
    ttl::tspan<int, std::extents<std::size_t, 4, 4>> _(a, 4, 4);
    ttl::tspan<int, std::extents<std::size_t, 2, 2, 2, 2>> _(a, 2, 2, 2, 2);

    ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent>> _(a, 16);
    ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>> _(a, 4, 4);
    ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>> _(a, 2, 2, 2, 2);

    // ttl::tspan _(b); // expected failure

    ttl::tspan _(a, std::extents<std::size_t, 16>());
    ttl::tspan _(a, std::extents<std::size_t, 4, 4>());
    ttl::tspan _(a, std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(a, 16);
    ttl::tspan _(a, 4, 4);
    ttl::tspan _(a, 2, 2, 2, 2);

    delete [] a;

    int const *b = new int const[16]{};
    // ttl::tspan _(b); // expected failure

    ttl::tspan _(b, std::extents<std::size_t, 16>());
    ttl::tspan _(b, std::extents<std::size_t, 4, 4>());
    ttl::tspan _(b, std::extents<std::size_t, 2, 2, 2, 2>());

    ttl::tspan _(b, 16);
    ttl::tspan _(b, 4, 4);
    ttl::tspan _(b, 2, 2, 2, 2);

    delete [] b;

    return true;
}

static_assert(check_c_array());
static_assert(check_array());
static_assert(check_vector());
static_assert(check_new());