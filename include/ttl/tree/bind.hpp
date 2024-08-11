#pragma once

#include <ttl/expression.hpp>
#include <ttl/extents.hpp>
#include <ttl/index.hpp>
#include <ttl/tensor.hpp>
#include <ttl/tree/node.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace ttl::tree
{
    template <tensor A, index_string _index>
    struct bind : node {
        static_assert(ttl::rank<A> == _index.size());

        static constexpr auto _outer = _index.outer();
        static constexpr auto _inner = _index.inner();
        static constexpr auto _all = _index.all();
        static constexpr auto _rank = _outer.rank();

        A _a;
        ttl::index<_index> _id;

        constexpr bind(A a, ttl::index<_index> id = {})
            : _a(a)
            , _id(id)
        {
            assert(_check_contracted_extents<_index>(ttl::extents(_a)));
        }

        constexpr auto operator=(scalar_type<A> const& x) -> bind&
            requires requires { ttl::evaluate(*this) = x; }
        {
            _evaluate() = x;
            return *this;
        }

        // Can't do this because of
        // https://github.com/llvm/llvm-project/issues/54440, must manually
        // compute the right type.
        //
        // constexpr operator decltype(_evaluate())() const
        constexpr operator std::conditional_t<_index.contracted().size() == 0, ttl::evaluate_type<A>, ttl::scalar_type<A>>() const
            requires(_rank == 0)
        {
            return _evaluate();
        }

        static constexpr auto outer()
        {
            return _outer;
        }

        constexpr auto extents() const
        {
            return select_extents(index_map<_index, _outer>, ttl::extents(_a));
        }

        constexpr auto operator[](this auto&& self, std::integral auto... i) -> decltype(__fwd(self)._evaluate(i...))
        {
            static_assert(sizeof...(i) == _rank);
            assert(self._check_bounds(i...));
            return __fwd(self)._evaluate(i...);
        }

    private:
        /// This innermost evaluate implementation finally forwards to _a.
        ///
        /// This is called once we have enough indices, i..., to satisfy all of
        /// the _inner extents. If this bind represents a contraction (i.e.,
        /// some sort of trace) then we need to duplicate and potentially
        /// shuffle some of the incoming indices.
        ///
        ///    int x[2][2]
        ///    {
        ///        {1, 0},
        ///        {0, 2}
        ///    };
        ///    auto b = bind(x, ii);
        ///    auto tr = b[]
        //         -> b._evaluate(0) + b._evaluate(1)
        ///        -> b._evaluate_impl(std::index_sequence<0,0>, 0) + b._evaluate(1)
        ///        -> ttl::evaluate(b.x, 0, 0) + b._evaluate(1)
        ///        -> 1 + b._evaluate(1)
        ///        -> 1 + b._evaluate_impl(std::index_sequence<0,0>, 1)
        ///        -> 1 + ttl::evaluate(b.x, 1, 1)
        ///        -> 1 + 2
        ///        -> 3
        template <std::size_t... j>
        constexpr auto _remap_indices(this auto&& self, std::index_sequence<j...>, std::integral auto... i) -> ttl::evaluate_type<A>
        {
            static_assert(sizeof...(i) == _all.size());
            int const ind[] { i... };
            return ttl::evaluate(__fwd(self)._a, ind[j]...);
            // return ttl::evaluate(__fwd(self)._a, i...[j]...); @todo[c++26]
        }

        /// This is called to append any projected indices to the pack.  It will
        /// forward to the _remap_indices to forward to remap the index pack to
        /// the order expected by the _a tensor space.
        template <std::size_t... j>
        constexpr auto _append_projection(this auto&& self, std::index_sequence<j...>, std::integral auto... i) -> ttl::evaluate_type<A>
        {
            return __fwd(self)._remap_indices(index_map<_all, _index>, i..., self._id[j]...);
        }

        /// This is called when we have enough indices, i..., to satisfy the
        /// _inner index. It will forward to the function that appends and
        /// projected indices.
        constexpr auto _evaluate(this auto&& self, std::integral auto... i) -> ttl::evaluate_type<A>
            requires(sizeof...(i) == _inner.size())
        {
            return __fwd(self)._append_projection(self._id.projection_map(), i...);
        }

        /// This is called when the bind represents a contraction, and we
        /// haven't generated enough indices for that contraction yet.
        constexpr auto _evaluate(this auto&& self, std::integral auto... i) -> ttl::scalar_type<A>
        {
            static_assert(_rank <= sizeof...(i) and sizeof...(i) < _inner.size());
            auto const extents = select_extents(index_map<_index, _inner>, ttl::extents(self._a));
            accumulator_type<A> accum {};
            for (std::size_t j = 0, e = extents.extent(sizeof...(i)); j < e; ++j) {
                accum += self._evaluate(i..., j);
            }
            return accum;
        }
    };
}
