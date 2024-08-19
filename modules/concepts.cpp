module;

#include <array>
#include <concepts>
#include <mdspan>
#include <ranges>
#include <utility>
#include <vector>

module ttl:concepts;

namespace ttl::concepts
{
    template <class T>
    concept integral = std::integral<std::decay_t<T>>;

    template <class T>
    concept floating_point = std::floating_point<std::decay_t<T>>;

    template <class T>
    concept c_array = std::ranges::contiguous_range<T> and std::is_bounded_array_v<std::remove_cvref_t<T>>;

    template <class T, std::size_t N>
    void check_std_array(std::array<T, N>&);

    template <class T, std::size_t N>
    void check_std_array(std::array<T, N> const&);

    template <class T>
    concept array = std::ranges::contiguous_range<T> and requires (T t) {
        check_std_array(t);
    };

    template <class T>
    void check_std_vector(std::vector<T>&);

    template <class T>
    void check_std_vector(std::vector<T> const&);

    template <class T>
    concept vector = std::ranges::contiguous_range<T> and requires (T t) {
        check_std_vector(t);
    };

    template <class T, std::size_t N>
    void check_std_span(std::span<T, N>&);

    template <class T, std::size_t N>
    void check_std_span(std::span<T, N> const&);

    template <class T>
    concept span = std::ranges::contiguous_range<T> and requires (T t) {
        check_std_span(t);
    };

    template <class T, class Extents, class Layout, class Accessor>
    void check_std_mdspan(std::mdspan<T, Extents, Layout, Accessor>&);

    template <class T, class Extents, class Layout, class Accessor>
    void check_std_mdspan(std::mdspan<T, Extents, Layout, Accessor> const&);

    template <class T>
    concept mdspan = requires (T t) {
        check_std_mdspan(t);
    };

    template <class>
    inline constexpr bool _check_std_extents = false;

    template <class T, std::size_t... Extents>
    inline constexpr bool _check_std_extents<std::extents<T, Extents...>> = true;

    template <class T>
    concept extents = _check_std_extents<std::decay_t<T>>;

    template <class>
    inline constexpr bool _check_integral_constant = false;

    template <class T, T N>
    inline constexpr bool _check_integral_constant<std::integral_constant<T, N>> = true;

    template <class T>
    concept integral_constant = _check_integral_constant<std::decay_t<T>>;

    template <class>
    inline constexpr bool _check_index_sequence = false;

    template <std::size_t... Is>
    inline constexpr bool _check_index_sequence<std::index_sequence<Is...>> = true;

    template <class T>
    concept index_sequence = _check_index_sequence<std::decay_t<T>>;
}
