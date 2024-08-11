#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace ttl
{
    template <class> struct tensor_traits;

    inline constexpr class _evaluate_fn
    {
        template <class T, std::size_t N>
        static constexpr bool _has_trait = []<std::size_t... i>(std::index_sequence<i...>) {
            return requires (T&& t) { tensor_traits<std::remove_reference_t<T>>::evaluate(__fwd(t), ((void)i, 0zu)...); };
        }(std::make_index_sequence<N>());

        template <class T, std::size_t N>
        static constexpr bool _has_member_fn = []<std::size_t... i>(std::index_sequence<i...>) {
            return requires (T&& t) { tensor_traits<std::remove_reference_t<T>>::evaluate(__fwd(t), ((void)i, 0zu)...); };
        }(std::make_index_sequence<N>());

        template <class T, std::size_t N>
        static constexpr bool _has_multidimensional_index = []<std::size_t... i>(std::index_sequence<i...>) {
            return requires (T&& t) { __fwd(t)[((void)i, 0zu)...]; };
        }(std::make_index_sequence<N>());

        template <class T, std::size_t N>
        static constexpr bool _use_trait = _has_trait<T, N>;

        template <class T, std::size_t N>
        static constexpr bool _use_member_fn = not _use_trait<T, N> and _has_member_fn<T, N>;

        template <class T, std::size_t N>
        static constexpr bool _use_multidimensional_index = not _use_trait<T, N> and not _use_member_fn<T, N> and _has_multidimensional_index<T, N>;

    public:
        template <class T>
        static constexpr auto operator()(T&& t, std::integral auto... i) -> decltype(tensor_traits<std::remove_reference_t<T>>::evaluate(__fwd(t), i...))
            requires _use_trait<T, sizeof...(i)>
        {
            return tensor_traits<std::remove_reference_t<T>>::evaluate(__fwd(t), i...);
        }

        template <class T>
        static constexpr auto operator()(T&& t, std::integral auto... i) -> decltype(__fwd(t).evaluate(i...))
            requires _use_member_fn<T, sizeof...(i)>
        {
            return __fwd(t).evaluate(i...);
        }

        template <class T>
        static constexpr auto operator()(T&& t, std::integral auto... i) -> decltype(__fwd(t)[i...])
            requires _use_multidimensional_index<T, sizeof...(i)>
        {
            return __fwd(t)[i...];
        }
    } evaluate;

    template <class T>
    using evaluate_type = decltype([]<std::size_t... i>(std::index_sequence<i...>) -> decltype(ttl::evaluate(std::declval<T>(), ((void)i, 0)...)) {
        return ttl::evaluate(std::declval<T>(), ((void)i, 0)...);
    }(std::make_index_sequence<rank<T>>()));

    template <class T>
    using scalar_type = std::remove_reference_t<evaluate_type<T>>;

    template <class T>
    using accumulator_type = std::remove_const_t<scalar_type<T>>;
}
