module;

#include <type_traits>

module ttl:outer;
import :index;
import :istring;
import :rank;

namespace ttl
{
    template <class T>
    concept _has_outer_member = requires (T&& t) {
        std::remove_cvref_t<T>::outer();
    };

    template <class T>
    struct _outer_impl;

    template <class T>
        requires (rank<T> == 0)
    struct _outer_impl<T> {
        static constexpr auto value = ttl::istring<1>{};
    };

    template <_has_outer_member T>
        requires (rank<T> != 0)
    struct _outer_impl<T> {
        static constexpr auto value = T::outer();
    };

    template <class T>
    concept _has_outer_impl = requires {
        _outer_impl<std::remove_cvref_t<T>>::value;
    };

    template <_has_outer_impl T>
    inline constexpr auto outer = _outer_impl<std::remove_cvref_t<T>>::value;

    namespace concepts
    {
        template <class T>
        concept has_outer = requires {
            outer<T>;
        };
    }
}
