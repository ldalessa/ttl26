#pragma once

#include <ttl/concepts.hpp>
#include <ttl/tensor.hpp>
#include <ttl/tensor_extents.hpp>
#include <ttl/tensor_index.hpp>
#include <ttl/tree/_index_mapper.hpp>

#include <cstddef>
#include <utility>

namespace ttl::tree
{
    template <concepts::expression A, concepts::expression B>
    struct sum
    {
        static constexpr auto _outer_a = ttl::outer<A>;
        static constexpr auto _outer_b = ttl::outer<B>;
        static constexpr auto _rank = _outer_a.rank();
        static constexpr auto _map = _index_mapper<_rank>(_outer_a, _outer_b);

        static_assert(_outer_a.is_permutation_of(_outer_b));

        A _a;
        B _b;

        constexpr auto extents() const
            TTL_ARROW ( merge_extents(_a.extents(), _b.extents()) );

        constexpr auto operator[](concepts::size_t auto... i)
            TTL_ARROW ( _evaluate(i...) );

      private:

        template <concepts::size_t... Is>
        requires (_rank == sizeof...(Is))
        constexpr auto _evaluate(Is... i) const
            TTL_ARROW ( _a[i...] + _map(_b, i...) );
    };
}

namespace ttl
{
    template <concepts::expression A, concepts::expression B>
    struct traits<tree::sum<A, B>>
    {
        using sum = tree::sum<A, B>;

        static consteval auto outer()
            TTL_ARROW ( sum::_outer_a );

        static constexpr auto tensor_extents(sum const& x)
            TTL_ARROW ( x.extents() );

        static constexpr auto evaluate(sum const& x, concepts::size_t auto... i)
            TTL_ARROW ( x._evaluate(std::size_t(i)...) );
    };

    template <concepts::expression A, concepts::expression B>
    constexpr auto operator+(A&& a, B&& b) -> tree::sum<A, B> {
        return tree::sum<A, B> {
            ._a = std::forward<A>(a),
            ._b = std::forward<B>(b)
        };
    }
}
