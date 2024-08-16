module;

#include <array>
#include <concepts>
#include <cstdint>
#include <mdspan>
#include <ranges>
#include <utility>
#include <vector>

export module ttl:tensor;
import :extents;

inline constexpr struct _extents_fn
{
    // static constexpr auto operator()(integral auto) -> std::extents<std::size_t> {
    //     return {};
    // }

    // static constexpr auto operator()(floating_point auto) -> std::extents<std::size_t> {
    //     return {};
    // }

    // template <carray T>
    // static constexpr std::size_t _extent = std::extent_v<std::remove_cvref_t<T>, 0>;

    // template <carray T>
    // constexpr auto operator()(this auto self, T&& t)
    //     -> decltype(prepend_extent<_extent<T>>(self(t[0])))
    // {
    //     return prepend_extent<_extent<T>>(self(t[0]));
    // }

    // template <class T, std::size_t N>
    // constexpr auto operator()(this auto self, std::span<T, N> s)
    //     -> decltype(prepend_extent<N>(self(s[0]), s.size()))
    // {
    //     return prepend_extent<N>(self(s[0]), s.size());
    // }

    // template <class T, std::size_t N>
    // static constexpr std::size_t _extent<std::array<T, N>> = N;

    // template <array T>
    // constexpr auto operator()(this auto self, array auto&& a)
    //     -> decltype(prepend_extent<_extent<std::decay_t<T>>>(self(a[0])))
    // {
    //     return prepend_extent<_extent<std::decay_t<T>>>(self(a[0]));
    // }

    // template <class T, class Extents, class L, class A>
    // static constexpr auto operator()(std::mdspan<T, Extents, L, A> const& s) -> Extents {
    //     return s.extents();
    // }

    // template <class T>
    // static constexpr auto operator()(T&& t) -> decltype(ttl::tensor_traits<std::remove_reference_t<T>>::extents(FWD(t)))
    //     requires requires { ttl::tensor_traits<std::remove_reference_t<T>>::extents(FWD(t)); }
    // {
    //     return ttl::tensor_traits<std::remove_reference_t<T>>::extents(FWD(t));
    // }
} extents;

// template <class T>
// inline constexpr std::size_t rank = decltype(auto(extents(std::declval<T>())))::rank();


// inline constexpr struct
// {
//     static constexpr auto operator()(integral auto && t) -> decltype(FWD(t)) {
//         return FWD(t);
//     }

//     static constexpr auto operator()(floating_point auto&& t) -> decltype(FWD(t)) {
//         return FWD(t);
//     }

//     constexpr auto operator()(this auto self, std::ranges::forward_range auto&& t, std::size_t i, std::integral auto... j)
//         -> decltype(self(*std::ranges::next(FWD(t), i), j...))
//     {
//         return self(*std::ranges::next(FWD(t), i), j...);
//     }

//     // constexpr auto operator()(this auto self, carray auto&& t, std::size_t i, std::integral auto... j)
//     //     -> decltype(self(FWD(t)[i], j...))
//     // {
//     //     return self(FWD(t)[i], j...);
//     // }

//     // template <class T, std::size_t N>
//     // constexpr auto operator()(this auto self, std::span<T, N> t, std::size_t i, std::integral auto... j)
//     //     -> decltype(self(FWD(t)[i], j...))
//     // {
//     //     return self(FWD(t)[i], j...);
//     // }

//     // constexpr auto operator()(this auto self, array auto&& t, std::size_t i, std::integral auto... j)
//     //     -> decltype(self(FWD(t)[i], j...))
//     // {
//     //     return self(FWD(t)[i], j...);
//     // }

//     // template <class T>
//     // constexpr auto operator()(this auto self, std::vector<T>& t, std::size_t i, std::integral auto... j) -> decltype(self(t[i], j...)) {
//     //     return self(t[i], j...);
//     // }

//     // template <class T>
//     // constexpr auto operator()(this auto self, std::vector<T> const& t, std::size_t i, std::integral auto... j) -> decltype(self(t[i], j...)) {
//     //     return self(t[i], j...);
//     // }

//     // template <class T>
//     // constexpr auto operator()(this auto self, std::vector<T>&& t, std::size_t i, std::integral auto... j) -> decltype(self(FWD(t)[i], j...)) {
//     //     return self(FWD(t)[i], j...);
//     // }

//     template <std::size_t N, std::size_t... i>
//     constexpr auto operator()(this auto self, auto&& t, std::array<std::size_t, N> const& index, std::index_sequence<i...>)
//         -> decltype(self(FWD(t), index[i]...))
//     {
//         return self(FWD(t), index[i]...);
//     }

//     template <std::size_t N>
//     constexpr auto operator()(this auto self, auto&& t, std::array<std::size_t, N> const& index)
//         -> decltype(self(FWD(t), index, std::make_index_sequence<N>()))
//     {
//         return self(FWD(t), index, std::make_index_sequence<N>());
//     }
// } evaluate;

// template <class T, std::size_t N>
// concept evaluate_with_n = requires (T&& t, std::array<std::size_t, rank<T>> const& index) {
//     evaluate(FWD(t), index);
// };

// static_assert(std::ranges::range<std::span<int, 1>>);
// static_assert(std::ranges::forward_range<std::span<int, 1>>);

// static_assert(evaluate_with_n<int, 0>);
// static_assert(evaluate_with_n<int[1], 1>);
// static_assert(evaluate_with_n<int[1][1], 2>);

// static_assert(evaluate_with_n<int&, 0>);
// static_assert(evaluate_with_n<int(&)[1], 1>);
// static_assert(evaluate_with_n<int(&)[1][1], 2>);

// static_assert(evaluate_with_n<int&&, 0>);
// static_assert(evaluate_with_n<int(&&)[1], 1>);
// static_assert(evaluate_with_n<int(&&)[1][1], 2>);

// static_assert(evaluate_with_n<int const, 0>);
// static_assert(evaluate_with_n<int const[1], 1>);
// static_assert(evaluate_with_n<int const[1][1], 2>);

// static_assert(evaluate_with_n<int const&, 0>);
// static_assert(evaluate_with_n<int const(&)[1], 1>);
// static_assert(evaluate_with_n<int const(&)[1][1], 2>);

// static_assert(evaluate_with_n<int const&&, 0>);
// static_assert(evaluate_with_n<int const(&&)[1], 1>);
// static_assert(evaluate_with_n<int const(&&)[1][1], 2>);

// // static_assert(evaluate_with_n<std::span<int, 1>&, 1>);
// // static_assert(evaluate_with_n<std::array<int[1], 1>, 2>);

// // template <class T>
// // concept tensor = requires (T&& t) {
// //     extents(FWD(t));
// //     requires evaluate_with_n<T, rank<T>>;
// // };

// // static_assert(tensor<int>);
// // static_assert(tensor<int&>);
// // static_assert(tensor<int&&>);
// // static_assert(tensor<int const>);
// // static_assert(tensor<int const&>);
// // static_assert(tensor<int const&&>);

// // static_assert(tensor<float>);
// // static_assert(tensor<float&>);
// // static_assert(tensor<float&&>);
// // static_assert(tensor<float const>);
// // static_assert(tensor<float const&>);
// // static_assert(tensor<float const&&>);

// // static_assert(tensor<int[1]>);
// // static_assert(tensor<int(&)[1]>);
// // static_assert(tensor<int(&&)[1]>);
// // static_assert(tensor<int const[1]>);
// // static_assert(tensor<int const(&)[1]>);
// // static_assert(tensor<int const(&&)[1]>);

// // static_assert(tensor<int[1][1]>);
// // static_assert(tensor<int(&)[1][1]>);
// // static_assert(tensor<int(&&)[1][1]>);
// // static_assert(tensor<int const[1][1]>);
// // static_assert(tensor<int const(&)[1][1]>);
// // static_assert(tensor<int const(&&)[1][1]>);

// // static_assert(tensor<std::span<int, 1>>);
// // static_assert(tensor<std::span<int const, 1>>);

// // // static_assert(tensor<std::array<int, 1>>);
// // // static_assert(tensor<std::array<int const, 1>>);

// // // template <class T, std::size_t N>
// // // concept tensor_of_rank = tensor<T> and rank<T> == N;

// // // static_assert(tensor_of_rank<int, 0>);
// // // static_assert(tensor_of_rank<int&, 0>);
// // // static_assert(tensor_of_rank<int&&, 0>);
// // // static_assert(tensor_of_rank<int const, 0>);
// // // static_assert(tensor_of_rank<int const&, 0>);
// // // static_assert(tensor_of_rank<int const&&, 0>);

// // // static_assert(tensor_of_rank<float, 0>);
// // // static_assert(tensor_of_rank<float&, 0>);
// // // static_assert(tensor_of_rank<float&&, 0>);
// // // static_assert(tensor_of_rank<float const, 0>);
// // // static_assert(tensor_of_rank<float const&, 0>);
// // // static_assert(tensor_of_rank<float const&&, 0>);


// // // static_assert(tensor<int[1]>);
// // // static_assert(tensor<int(&)[1]>);
// // // static_assert(tensor_of_rank<int(&)[1], 1>);
// // // static_assert(tensor_of_rank<int(&&[1]), 1>);
// // // static_assert(tensor_of_rank<int const[1], 1>);
// // // static_assert(tensor_of_rank<int const(&)[1], 1>);
// // // static_assert(tensor_of_rank<int const(&&)[1], 1>);
