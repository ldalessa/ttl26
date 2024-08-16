module;

#include <cstddef>

export module ttl:index;
import :istring;

/// Strong typedef for indexes.
///
/// Right now this is strongly typdeffed to support u16 characters
/// which is good enough for the set of normal ascii and greek
/// characters that are commonly used in tensor indexing.
///
/// In the future this could be set by the build system.
///
/// It is given the name `i` and not part of the ttl namespace in 
/// order to minimize the amount of characters that are printed
/// during compilation errors. The template class is not exported
/// so it shouldn't collide with anything. The deduction guide *is*
/// exported so that `ttl::index<"i"> i;` can properly deduce.
template <std::size_t N>
struct i : istring<N>
{
    using i::istring::istring;
};

export template <character T, std::size_t N>
i(T const(&)[N]) -> i<N>;

export namespace ttl
{
    /// The ttl::index serves as a CNTTP wrapper so that we
    /// can process tensor expressions using constexpr index
    /// declarations. They also serve to keep track of any
    /// index values for projected indices (not unlike how 
    /// std::extent will track extents for dynamic extents).
    template <i str>
    struct index
    {
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
    template <i a, i b>
    inline constexpr auto operator+(index<a> const&, index<b> const&) 
        -> index<a + b> 
    {
        return {};
    }

    namespace literals
    {
        template <i str>
        inline consteval auto operator""_i() -> index<str> {
            return {};
        }
    }
}

#undef NDEBUG

static consteval bool test_index() {
    using namespace ttl::literals;
    i n = "n";
    i m = "m";
    (void)(n + m);

    ttl::index<"i"> i;
    auto j = "j"_i;
    ttl::index p(1);

    (void)(i + j + p);

    return true;
}

static_assert(test_index());