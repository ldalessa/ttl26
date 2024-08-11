#pragma once

#include <ttl/evaluate.hpp>
#include <ttl/expression.hpp>
#include <ttl/index.hpp>
#include <ttl/outer.hpp>
#include <ttl/tree/node.hpp>

#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>

namespace ttl::tree
{
    template <expression A, expression B, auto op, auto reduce>
    struct product : node
    {
        using scalar_type = std::invoke_result_t<decltype(op), scalar_type<A>, scalar_type<B>>;
        using accumulator_type = std::remove_cvref_t<scalar_type>;

        static constexpr auto _outer_a = ttl::outer<A>;
        static constexpr auto _outer_b = ttl::outer<B>;
        static constexpr auto _outer_ab = _outer_a + _outer_b;

        static_assert(_outer_ab.projected().size() == 0);

        static constexpr auto _outer = _outer_ab.outer();
        static constexpr auto _inner = _outer_ab.inner();
        static constexpr auto _rank = _outer.rank();

        /// Index maps for the inner evaluate.
        static constexpr auto _map_a = index_map<_inner, _outer_a>;
        static constexpr auto _map_b = index_map<_inner, _outer_b>;

        A _a;
        B _b;

        constexpr product(A a, B b)
            : _a(__fwd(a))
            , _b(__fwd(b))
        {
            assert(_check_contracted_extents<_outer_ab>(_extents_ab()));
        }

        static constexpr auto outer() {
            return _outer;
        }

        constexpr operator scalar_type() const
            requires (_rank == 0)
        {
            return _evaluate();
        }

        /// Select the extents from the a-b concatenated extents
        /// that correspond to extents that are exposed in the
        /// contraction's outer index space.
        constexpr auto extents() const
        {
            static constexpr auto map = index_map<_outer_ab, _outer>;
            auto const ab = _extents_ab();
            return select_extents(map, ab);
        }

        /// Evaluate the contraction for an index.
        constexpr auto operator[](std::integral auto... i) const -> scalar_type
        {
            static_assert(sizeof...(i) == _rank);
            assert(_check_bounds(i...));
            return _evaluate(i...);
        }

    private:
        /// Map the indices from i... into the outer space
        /// for A and B, evaluate both subexpressions, and
        /// combine them using the configured `op`.
        template <std::size_t... a, std::size_t... b>
        constexpr auto _evaluate(std::index_sequence<a...>, std::index_sequence<b...>, std::integral auto... i) const -> scalar_type
        {
            static_assert(sizeof...(i) == _inner.size());

            int const ind[]{ i... };
            return op(ttl::evaluate(_a, ind[a]...), ttl::evaluate(_b, ind[b]...));
            // return op(ttl::evaluate(_a, i...[a]...), ttl::evaluate(_b, i...[b]...)); @todo[c++26]
        }

        /// Forward to the inner evaluation, passing the
        /// appropriate maps.
        constexpr auto _evaluate(std::integral auto... i) const -> scalar_type
            requires (sizeof...(i) == _inner.size())
        {
            return _evaluate(_map_a, _map_b, i...);
        }

        /// Continue to expand and accumulate the contracted
        /// extents until we have enough indices to evaluate the
        /// inner expression.
        ///
        /// If we encounter this function then we have an
        /// accumulation occurring and thus we'll be returning the
        /// scalar_type.
        constexpr auto _evaluate(std::integral auto... i) const -> scalar_type
        {
            static constexpr auto N = sizeof...(i);
            static_assert(_rank <= N and N < _inner.size());

            // Map the extents from the concantenated extents for a.b
            // into the inner index space (the outer indices + the
            // contracted indices). Technically we only need the extents
            // for the contracted indices but it makes the loop a bit
            // easier to express if we also have the outer extents.
            static constexpr auto map = index_map<_outer_ab, _inner>;
            auto const ab = _extents_ab();
            auto const inner = select_extents(map, ab);

            // Accumulate the Nth extent. Help the compiler out here.
            if constexpr (inner.static_extent(N) == std::dynamic_extent) {
                accumulator_type accum{};
                for (std::size_t j = 0, e = inner.extent(N); j != e; ++j) {
                    accum = reduce(accum, _evaluate(i..., j));
                }
                return accum;
            }
            else {
                static constexpr std::size_t e = inner.static_extent(N);
                accumulator_type accum{};
                for (std::size_t j = 0; j != e; ++j) {
                    accum = reduce(accum, _evaluate(i..., j));
                }
                return accum;
            }
        }

        /// Create the concatenated extents of the two subexpressions.
        constexpr auto _extents_ab() const {
            auto const a = ttl::extents(_a);
            auto const b = ttl::extents(_b);
            return concat_extents(a, b);
        }
    };

    template <expression A, expression B>
    struct mul : product<A, B, std::multiplies{}, std::plus{}> {
        using mul::product::product;
    };

    template <expression A, expression B>
    constexpr auto operator*(A&& a, B&& b) -> mul<A, B> {
        return mul<A, B>(__fwd(a), __fwd(b));
    }
}
