#pragma once

#include <cstddef>
#include <mdspan>
#include <type_traits>
#include <utility>

namespace ttl
{
    template <class> struct tensor_traits;

    inline constexpr class _extents_fn
    {
        template <class T>
        static constexpr bool _has_trait = requires (T&& t) {
            tensor_traits<std::remove_reference_t<T>>::extents(t);
        };

        template <class T>
        static constexpr bool _has_member_fn = requires (T&& t) {
            t.extents();
        };

        template <class T>
        static constexpr bool _use_trait = _has_trait<T>;

        template <class T>
        static constexpr bool _use_member_fn = not _use_trait<T> and _has_member_fn<T>;

    public:
        template <class T>
        static constexpr auto operator()(T&& t) -> decltype(tensor_traits<std::remove_reference_t<T>>::extents(t))
            requires _use_trait<T>
        {
            return tensor_traits<std::remove_reference_t<T>>::extents((T&&)t);
        }

        template <class T>
        static constexpr auto operator()(T&& t) -> decltype(t.extents())
            requires _use_member_fn<T>
        {
            return __fwd(t).extents();
        }
    } extents;

    template <class T>
    using extents_type = decltype(auto(ttl::extents(std::declval<T>())));

    template <class T>
    inline constexpr std::size_t rank = extents_type<T>::rank();

    template <class T>
    inline constexpr auto extent(T&& t, std::size_t i) -> std::size_t {
        return ttl::extents((T&&)t).extent(i);
    }

    template <std::size_t i, class T>
    inline constexpr auto extent(T&& t) -> std::size_t {
        return ttl::extent(__fwd(t), i);
    }

    template <std::size_t i, class T>
    inline constexpr auto static_extent = extents_type<T>::static_extent(i);

    template <class T, std::size_t... as, class U, std::size_t... bs>
    inline constexpr auto concat_extents(std::extents<T, as...> const& a, std::extents<U, bs...> const& b) -> std::extents<std::common_type_t<T>, as..., bs...> {
        return [&]<std::size_t... i, std::size_t... j>(std::index_sequence<i...>, std::index_sequence<j...>) {
            return std::extents<std::common_type_t<T, U>, as..., bs...> { a.extent(i)..., b.extent(j)... };
        }(std::make_index_sequence<sizeof...(as)>(), std::make_index_sequence<sizeof...(bs)>());
    }

    template <std::size_t a, class T, std::size_t... bs>
    inline constexpr auto prepend_extent(std::extents<T, bs...> const& b, T x) {
        return concat_extents(std::extents<T, a>(x), b);
    }

    template <std::size_t... i, class T, std::size_t... ts>
    inline constexpr auto select_extents(std::extents<T, ts...> const& t) {
        return std::extents<T, std::extents<T, ts...>::static_extent(i)...> {
            t.extent(i)...
        };
    }

    template <std::size_t... i, class T, std::size_t... ts>
    inline constexpr auto select_extents(std::index_sequence<i...> const&, std::extents<T, ts...> const& t)
    {
        return select_extents<i...>(t);
    }

    template <class T, std::size_t... as, class U, std::size_t... bs>
    inline constexpr bool compatible_extents(std::extents<T, as...> const& a, std::extents<U, bs...> const& b)
    {
        static constexpr std::size_t N = sizeof...(as);
        static constexpr std::size_t M = sizeof...(bs);
        static_assert(N == M);
        static_assert(
            ((as == bs or as == std::dynamic_extent or bs == std::dynamic_extent) && ...),
            "Extents are incompatible.");

        return [&]<std::size_t... i>(std::index_sequence<i...>) {
            return ((a.extent(i) == b.extent(i)) && ...);
        }(std::make_index_sequence<N>());
    }

    template <class T, std::size_t... as, class U, std::size_t... bs>
    inline constexpr auto merge_extents(std::extents<T, as...> const& a, std::extents<U, bs...> const& b)
        -> std::extents<std::common_type_t<T, U>, std::min(as, bs)...>
    {
        assert(compatible_extents(a, b));
        return std::extents<std::common_type_t<T, U>, std::min(as, bs)...>{a};
    }
}
