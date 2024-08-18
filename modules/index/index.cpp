module;

#include <cstddef>
#include <type_traits>

export module ttl:index;
import :istring;

namespace ttl
{
    /// The ttl::index serves as a CNTTP wrapper so that we
    /// can process tensor expressions using constexpr index
    /// declarations. They also serve to keep track of any
    /// index values for projected indices (not unlike how
    /// std::extent will track extents for dynamic extents).
    export template <istring str>
    struct index
    {
        using _ttl_index_tag_type = void;

        static constexpr auto _size = str.size();
        int _projected[_size]{};

        consteval index() = default;

        /// This constructor is used to initialize a projected
        /// extent.
        constexpr index(int p) : _projected{p} {
            static_assert(_size == 1 and str[0] == str.projected);
        }
    };

    index(int) -> index<projection>;

    /// Clients can concatentate their indices with operator+.
    template <istring a, istring b>
    inline constexpr auto operator+(index<a> const&, index<b> const&)
        -> index<a + b>
    {
        return {};
    }

    export namespace literals
    {
        template <istring str>
        inline consteval auto operator""_i() -> index<str> {
            return {};
        }
    }

    namespace concepts
    {
        template <class T>
        concept index = requires {
            typename std::decay_t<T>::_ttl_index_tag_type;
        };
    }
}

#undef NDEBUG

static consteval bool test_index() {
    using namespace ttl::literals;
    ttl::istring n = "n";
    ttl::istring m = "m";
    (void)(n + m);

    ttl::index<"i"> i;
    auto j = "j"_i;
    ttl::index p(1);

    (void)(i + j + p);

    return true;
}

static_assert(test_index());
