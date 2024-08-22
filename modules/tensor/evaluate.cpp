module;

#include <array>
#include <cassert>
#include <concepts>
#include <mdspan>
#include <ranges>
#include <type_traits>
#include <utility>

module ttl:evaluate;
import :concepts;
import :rank;
import :tensor_traits;

namespace stdr = std::ranges;

namespace ttl
{
    /// Implement the evaluate() overload set as a function object.
    inline constexpr struct _evaluate_fn
    {
        /// Evaluate a stdlib scalar type.
        template <class T>
            requires (concepts::integral<T> or concepts::floating_point<T>)
        static constexpr auto operator()(T&& t) ->
            ARROW( FWD(t) );

        /// Evaluate any type that has the tensr_trait::evaluate defined.
        template <class T, std::integral... Is>
            requires concepts::has_evaluate_trait<T, Is...>
        static constexpr auto operator()(T&& t, Is... i) ->
            ARROW( ttl::tensor_traits<std::decay_t<T>>::evaluate(FWD(t), i...) );

        /// Evaluate any forward range.
        ///
        /// This simply pops a single index out of the index pack and uses it to
        /// index into the outermost range, forwarding the result to a recursive
        /// instantiation of evaluate().
        template <class T, std::integral... Is>
            requires (not concepts::has_evaluate_trait<T, std::size_t, Is...> and stdr::forward_range<T>)
        constexpr auto operator()(this auto const& self, T&& t, std::size_t i, Is... j) ->
            ARROW( self(*stdr::next(stdr::begin(t), i), j...) );

        /// Evaluate types with appropriate multidimensional index opeartors.
        ///
        /// This will match mdspan, but it will also match all of the expression
        /// tree types in the tree modules.
        template <class T, std::integral... Is>
            requires (not concepts::has_evaluate_trait<T, Is...> and not stdr::forward_range<T>)
        static constexpr auto operator()(T&& t, Is... i) ->
            ARROW( FWD(t)[i...] );

        /// The _check_n functions are here to help the has_evaluate_n concept.
        ///
        /// @{
        template <std::size_t N, std::size_t... i>
        constexpr auto _check_2(this auto const& self, auto&& t, std::array<std::size_t, N> const& index, std::index_sequence<i...>) ->
            ARROW( self(FWD(t), index[i]...) );

        template <std::size_t N>
        constexpr auto _check_1(this _evaluate_fn const& self, auto&& t, std::array<std::size_t, N> const& index) ->
            ARROW( self._check_2(FWD(t), index, std::make_index_sequence<N>()) );

        template <class T>
        constexpr auto _check_0(this _evaluate_fn const& self, T&& t) ->
            ARROW ( self._check_1(FWD(t), std::array<std::size_t, rank<T>>()) );
        /// @}
    } evaluate;

    /// Get the type of ttl::evaluate(T).
    template <class T>
    using evaluate_type = decltype(evaluate._check_0(std::declval<T>()));

    /// Get the scalar type for a tensor.
    ///
    /// The scalar type is the value type of the result of ttl::evaluate. The
    /// default type is simply the remove_cvref_t of the result of calling
    /// evaluate, which will always work, however to reduce the compile time
    /// cost we allow either the tensor_traits or tensor class to specialize
    /// scalar_type.
    ///
    /// @{
    namespace _scalar_type
    {
        template <class T>
        concept use_tensor_trait = requires {
            typename tensor_traits<std::decay_t<T>>::scalar_type;
        };

        template <class T>
        concept use_member = not use_tensor_trait<T> and requires {
            typename std::decay_t<T>::scalar_type;
        };

        template <class T>
        struct impl {
            using type = std::remove_cvref_t<evaluate_type<T>>;
        };

        template <use_tensor_trait T>
        struct impl<T> {
            using type = tensor_traits<std::decay_t<T>>::scalar_type;
        };

        template <use_member T>
        struct impl<T> {
            using type = std::decay_t<T>::scalar_type;
        };
    }

    template <class T>
    using scalar_type = _scalar_type::impl<T>::type;
    /// @}

    namespace concepts
    {
        template <class T, std::size_t N>
        concept has_evaluate_n = requires (T&& t) {
            evaluate._check_0(FWD(t));
        };
    }
}

using namespace ttl;
using namespace ttl::concepts;

#undef DNDEBUG

static_assert(has_evaluate_n<float, 0>);

static_assert(has_evaluate_n<int, 0>);
static_assert(has_evaluate_n<int[1], 1>);
static_assert(has_evaluate_n<int[1][1], 2>);

static_assert(has_evaluate_n<int&, 0>);
static_assert(has_evaluate_n<int(&)[1], 1>);
static_assert(has_evaluate_n<int(&)[1][1], 2>);

static_assert(has_evaluate_n<int&&, 0>);
static_assert(has_evaluate_n<int(&&)[1], 1>);
static_assert(has_evaluate_n<int(&&)[1][1], 2>);

static_assert(has_evaluate_n<int const, 0>);
static_assert(has_evaluate_n<int const[1], 1>);
static_assert(has_evaluate_n<int const[1][1], 2>);

static_assert(has_evaluate_n<int const&, 0>);
static_assert(has_evaluate_n<int const(&)[1], 1>);
static_assert(has_evaluate_n<int const(&)[1][1], 2>);

static_assert(has_evaluate_n<int const&&, 0>);
static_assert(has_evaluate_n<int const(&&)[1], 1>);
static_assert(has_evaluate_n<int const(&&)[1][1], 2>);

static_assert(has_evaluate_n<std::span<int, 1>, 1>);
static_assert(has_evaluate_n<std::span<int[1], 1>, 2>);
static_assert(has_evaluate_n<std::span<int[1][1], 1>, 3>);

static constexpr bool check_evaluate()
{
    int a = 42;
    assert(evaluate(a) == 42);
    assert(evaluate(std::as_const(a)) == 42);
    assert(evaluate(std::move(a)) == 42);
    assert((evaluate(a) = 1) == 1);

    evaluate((int[1]){1}, 0);

    int b[] = { 1, 2 };
    assert(evaluate(std::span(b), 0) == 1);
    assert(evaluate(std::span(b), 1) == 2);
    assert(evaluate(std::span(std::as_const(b)), 0) == 1);
    assert(evaluate(std::span(std::as_const(b)), 1) == 2);
    assert(evaluate(std::mdspan(b, 2, 1), 0, 0) == 1);
    assert(evaluate(std::mdspan(b, 2, 1), 0, 1) == 2);
    assert(evaluate(std::mdspan(std::as_const(b), 2, 1), 0, 0) == 1);
    assert(evaluate(std::mdspan(std::as_const(b), 2, 1), 0, 1) == 2);
    assert(evaluate(b, 0) == 1);
    assert(evaluate(b, 1) == 2);
    assert(evaluate(std::as_const(b), 0) == 1);
    assert(evaluate(std::as_const(b), 1) == 2);
    assert(evaluate(std::move(b), 0) == 1);
    assert(evaluate(std::move(b), 1) == 2);

    evaluate(b, 0) = 3;
    evaluate(b, 1) = 4;
    assert(b[0] == 3);
    assert(b[1] == 4);

    evaluate(std::span(b), 0) = 5;
    evaluate(std::span(b), 1) = 6;
    assert(b[0] == 5);
    assert(b[1] == 6);

    evaluate(std::mdspan(b, 2, 1), 0, 0) = 7;
    evaluate(std::mdspan(b, 2, 1), 0, 1) = 8;
    assert(b[0] == 7);
    assert(b[1] == 8);

    int c[][2] = { {1, 2}, {3, 4} };
    assert(evaluate(c, 0, 0) == 1);
    assert(evaluate(c, 0, 1) == 2);
    assert(evaluate(c, 1, 0) == 3);
    assert(evaluate(c, 1, 1) == 4);

    assert(evaluate(std::as_const(c), 0, 0) == 1);
    assert(evaluate(std::as_const(c), 0, 1) == 2);
    assert(evaluate(std::as_const(c), 1, 0) == 3);
    assert(evaluate(std::as_const(c), 1, 1) == 4);

    assert(evaluate(std::move(c), 0, 0) == 1);
    assert(evaluate(std::move(c), 0, 1) == 2);
    assert(evaluate(std::move(c), 1, 0) == 3);
    assert(evaluate(std::move(c), 1, 1) == 4);

    evaluate(c, 0, 0) = 5;
    evaluate(c, 0, 1) = 6;
    evaluate(c, 1, 0) = 7;
    evaluate(c, 1, 1) = 8;

    assert(c[0][0] == 5);
    assert(c[0][1] == 6);
    assert(c[1][0] == 7);
    assert(c[1][1] == 8);

    std::array d = { 1, 2 };
    assert(evaluate(d, 0) == 1);
    assert(evaluate(d, 1) == 2);
    assert(evaluate(std::as_const(d), 0) == 1);
    assert(evaluate(std::as_const(d), 1) == 2);
    assert(evaluate(std::move(d), 0) == 1);
    assert(evaluate(std::move(d), 1) == 2);

    evaluate(d, 0) = 3;
    evaluate(d, 1) = 4;
    assert(d[0] == 3);
    assert(d[1] == 4);

    std::array e = { std::array{1, 2}, std::array{3, 4} };
    assert(evaluate(e, 0, 0) == 1);
    assert(evaluate(e, 0, 1) == 2);
    assert(evaluate(e, 1, 0) == 3);
    assert(evaluate(e, 1, 1) == 4);

    assert(evaluate(std::as_const(e), 0, 0) == 1);
    assert(evaluate(std::as_const(e), 0, 1) == 2);
    assert(evaluate(std::as_const(e), 1, 0) == 3);
    assert(evaluate(std::as_const(e), 1, 1) == 4);

    assert(evaluate(std::move(e), 0, 0) == 1);
    assert(evaluate(std::move(e), 0, 1) == 2);
    assert(evaluate(std::move(e), 1, 0) == 3);
    assert(evaluate(std::move(e), 1, 1) == 4);

    evaluate(e, 0, 0) = 5;
    evaluate(e, 0, 1) = 6;
    evaluate(e, 1, 0) = 7;
    evaluate(e, 1, 1) = 8;

    assert(e[0][0] == 5);
    assert(e[0][1] == 6);
    assert(e[1][0] == 7);
    assert(e[1][1] == 8);

    return true;
}

static_assert(check_evaluate());
