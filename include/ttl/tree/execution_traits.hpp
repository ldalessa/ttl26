#pragma once

#include <ttl/bind.hpp>
#include <ttl/extents.hpp>
#include <ttl/evaluate.hpp>
#include <ttl/outer.hpp>
#include <ttl/tensor.hpp>
#include <ttl/tree/bind.hpp>

namespace ttl::tree
{
    template <tensor A, tensor B>
    struct execution_traits
    {
        static_assert(rank<A> == rank<B>);

        static constexpr auto assign(A&& a, B&& b) -> decltype(a)
        {
            assert(compatible_extents(extents(a), extents(b)));
            _assign(a, __fwd(b));
            return a;
        }

    private:
        /// Perform the inner contraction when A and B are expressions.
        template <std::size_t... i, std::size_t... j, std::integral... Ks>
        static constexpr void _assign(A& a, B const& b, std::index_sequence<i...>, std::index_sequence<j...>, Ks... k)
        {
            // return evaluate(a, k...[i]...), evaluate(b, k...[j]...)); @todo[c++26]
            std::common_type_t<Ks...> const ks[] { k... };
            evaluate(a, ks[i]...) = evaluate(b, ks[j]...);
        }

        static constexpr void _assign(A& a, B const& b, std::integral auto... i)
        {
            static constexpr auto N = sizeof...(i);
            
            if constexpr (N == rank<A>) {
                if constexpr (expression<A> and expression<B>) {
                    // If we have two expressions then make sure to handle any index
                    // remapping necessary.
                    static constexpr auto _outer_a = outer<A>;
                    static constexpr auto _outer_b = outer<B>;
                    static_assert(is_permutation(_outer_a, _outer_b));

                    static constexpr auto _map_a = index_map<_outer_a, _outer_a>;
                    static constexpr auto _map_b = index_map<_outer_a, _outer_b>;
                    _assign(a, b, _map_a, _map_b, i...);
                }
                else {
                    // If either one or neither of the arguments are expressions
                    // then we don't need to do any index remapping.
                    evaluate(a, i...) = evaluate(b, i...);
                }
            }
            // @todo: could optimize this case.
            // else if constexpr (static_extent<N, A> != std::dynamic_extent) {
            // }
            else {
                // We have a free index that we need to remap.
                auto const e = extent<N>(a);
                for (auto j = 0zu; j != e; ++j) {
                    _assign(a, b, i..., j);
                }
            }
        }
    };

    /// Bypass all the nonsense when we just have two scalars.
    template <scalar A, scalar B>
    struct execution_traits<A, B>
    {
        static constexpr auto assign(A a, B b) -> decltype(evaluate(__fwd(a)) = evaluate(__fwd(b)))
        {
            return evaluate(__fwd(a)) = evaluate(__fwd(b));
        }
    };
}