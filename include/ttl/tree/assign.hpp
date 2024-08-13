#pragma once

#include <ttl/tree/execution_traits.hpp>

namespace ttl::tree
{
    template <tensor A, tensor B>
    inline constexpr auto assign(A&& a, B&& b) -> decltype(execution_traits<A, B>::assign(__fwd(a), __fwd(b)))
    {
        return execution_traits<A, B>::assign(__fwd(a), __fwd(b));
    }

    template <tensor A, tensor B>
    inline constexpr auto operator<<(A&& a, B&& b) -> decltype(assign(__fwd(a), __fwd(b))) {
        static_assert(rank<A> == rank<B>);
        return assign(__fwd(a), __fwd(b));
    }

    template <tensor A, tensor B>
    inline constexpr auto operator+=(A&& a, B&& b) -> decltype(a) {
        a = __fwd(a) + __fwd(b);
        return a;
    }

    template <tensor A, tensor B>
    inline constexpr auto operator-=(A&& a, B&& b) -> decltype(a) {
        a = __fwd(a) - __fwd(b);
        return a;
    }

    template <scalar A, scalar B>
    inline constexpr auto operator*=(A&& a, B&& b)  -> decltype(a) {
        a = __fwd(a) * __fwd(b);
        return a;
    }
}