module;

#include <concepts>
#include <mdspan>
#include <type_traits>

export module ttl:tensor_traits;
import :concepts;

namespace ttl
{
    /// Tensor traits allow 3rd party types to be used as tensors.
    export template <class>
    struct tensor_traits;
    /// {
    ///     @required
    ///     static contexpr auto extents(T&&) -> concepts::extents;
    ///
    ///     @required
    ///     static contsexpr auto evaluate(T&&, std::integral auto...) -> scalar(&)
    ///
    ///     @optional
    ///     static constexpr auto rank() -> std::size_t
    ///
    ///     @optional
    ///     using extents_type = ...;
    ///
    ///     @optional
    ///     using scalar_type = ...;
    /// };
    namespace concepts
    {
        template <class T>
        concept has_extents_trait = requires (T&& t) {
            { tensor_traits<std::decay_t<T>>::extents(FWD(t)) } -> extents;
        };

        template <class T, class... I>
        concept has_evaluate_trait = requires (T&& t, I... i) {
            tensor_traits<std::decay_t<T>>::evaluate(FWD(t), i...);
        };

        template <class T>
        concept has_rank_trait = requires {
            { tensor_traits<std::decay_t<T>>::rank } -> concepts::integral_constant;
        };
    }
}

using namespace ttl::concepts;

namespace
{
    struct S {};
}

template <>
struct ttl::tensor_traits<S> {
    static constexpr auto extents(S&&) -> std::extents<std::size_t, 1> {
        return {};
    }

    static constexpr auto evaluate(S&&, int) -> int {
        return 0;
    }

    static constexpr std::integral_constant<std::size_t, 1> rank;
};

static_assert(has_extents_trait<S>);
static_assert(has_evaluate_trait<S, int>);
static_assert(has_rank_trait<S>);
