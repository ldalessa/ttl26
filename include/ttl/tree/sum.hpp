#pragma once

#include <ttl/evaluate.hpp>
#include <ttl/expression.hpp>
#include <ttl/index.hpp>
#include <ttl/outer.hpp>
#include <ttl/tree/node.hpp>

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace ttl::tree
{
    /// A generic sum node, will apply op(A, B).
    ///
    /// Concrete classes `add` and `sub` follow.
    ///
    /// The sum supports any sort of restructuring of indices as
    /// long as the extents match. For example A(i,j) + A(j,i)
    /// represents A + A^T.
    template <expression A, expression B, auto op>
    struct sum : node
    {
        using scalar_type = std::invoke_result_t<decltype(op), scalar_type<A>, scalar_type<B>>;

        static constexpr auto _outer_a = ttl::outer<A>;
        static constexpr auto _outer_b = ttl::outer<B>;
        static_assert(is_permutation(_outer_a, _outer_b));

        static constexpr auto _map_aa = index_map<_outer_a, _outer_a>;
        static constexpr auto _map_ab = index_map<_outer_a, _outer_b>;
        static constexpr auto _map_ba = index_map<_outer_b, _outer_a>;

        static constexpr auto _rank = _outer_a.rank();

        A _a;
        B _b;

        constexpr sum(A a, B b)
            : _a(__fwd(a))
            , _b(__fwd(b))
        {
            auto const extents_a = select_extents(_map_aa, ttl::extents(_a));
            auto const extents_b = select_extents(_map_ba, ttl::extents(_b));
            assert(compatible_extents(extents_a, extents_b));
        }

        static constexpr auto outer() {
            return _outer_a;
        }

        constexpr operator scalar_type(this sum const& self)
            requires (_rank == 0)
        {
            return self[];
        }

        constexpr auto extents() const {
            auto const extents_a = select_extents(_map_aa, ttl::extents(_a));
            auto const extents_b = select_extents(_map_ba, ttl::extents(_b));
            return merge_extents(extents_a, extents_b);
        }

        constexpr auto operator[](std::integral auto... i) const -> scalar_type
        {
            static_assert(sizeof...(i) == _rank);
            assert(_check_bounds(i...));
            return op(_evaluate(_a, _map_aa, i...), _evaluate(_b, _map_ab, i...));
        }

    private:
        template <std::size_t... i>
        static constexpr auto _evaluate(auto&& x, std::index_sequence<i...>, std::integral auto... j)
        {
            int const ind[]{j...};
            return ttl::evaluate(__fwd(x), ind[i]...);
            // return ttl::evaluate(__fwd(x), j...[i]...); @todo[c++26]
        }
    };

    template <expression A, expression B>
    struct add : sum<A, B, std::plus{}> {
        using add::sum::sum;
    };

    template <expression A, expression B>
    struct sub : sum<A, B, std::minus{}> {
        using sub::sum::sum;
    };

    template <expression A, expression B>
    inline constexpr auto operator+(A&& a, B&& b) -> add<A, B> {
        return add<A, B>(__fwd(a), __fwd(b));
    }

    template <expression A, expression B>
    inline constexpr auto operator-(A&& a, B&& b) -> sub<A, B> {
        return sub<A, B>(__fwd(a), __fwd(b));
    }
}
