module;

#include <type_traits>

module ttl:outer;
import :index;
import :rank;

namespace
{
    template <class T>
    concept has_outer_member = requires (T&& t) {
        { std::remove_cvref_t<T>::outer() } -> ttl_istring;
    };

    template <class T>
    struct outer_impl;

    template <class T>
    requires (rank<T> == 0)
    struct outer_impl<T> {
        static constexpr ttl_istring auto value = i<1>{};
    };

    template <has_outer_member T>
    requires (rank<T> != 0)
    struct outer_impl<T> {
        static constexpr ttl_istring auto value = T::outer();
    };

    template <class T>
    concept has_outer_impl = requires {
        { outer_impl<std::remove_cvref_t<T>>::value } -> ttl_istring;
    };
}

template <has_outer_impl T>
inline constexpr ttl_istring auto outer = outer_impl<std::remove_cvref_t<T>>::value;

template <class T>
concept has_outer = requires {
    { outer<T> } -> ttl_istring;
};
