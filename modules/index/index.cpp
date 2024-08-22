module;

#include <array>
#include <cassert>
#include <cstddef>
#include <ranges>
#include <type_traits>

export module ttl:index;
import :istring;

namespace stdr = std::ranges;

namespace ttl
{
    /// The ttl::index serves as a CNTTP wrapper so that we can process tensor
    /// expressions using constexpr index declarations. They also serve to keep
    /// track of any index values for projected indices (not unlike how
    /// std::extent will track extents for dynamic extents).
    export template <istring str>
    struct index
    {
        using _ttl_index_tag_type = void;

        static constexpr auto _size = str.size();
        std::array<std::size_t, _size> _projected{};

        consteval index() = default;

        /// This constructor is used to initialize a projected
        /// extent.
        constexpr index(std::size_t p) : _projected{p} {
            static_assert(_size == 1 and str[0] == str.projected);
        }

        static constexpr auto istring() {
            return str;
        }

        constexpr auto begin(this auto&& self) {
            return std::ranges::begin(self._projected);
        }

        constexpr auto end(this auto&& self) {
            return std::ranges::end(self._projected);
        }

        constexpr auto operator[](std::size_t i) const -> int {
            assert(i < _size);
            return _projected[i];
        }

        /// Create a projection map, which are the indices that should be
        /// selected from the underlying _projected array during evaluation.
        constexpr auto projection_map() const {
            return to_sequence<index_of<str, str.projected>>;
        }
    };

    index(int) -> index<projection>;

    /// Clients can concatentate their indices with operator+.
    template <istring a, istring b>
    inline constexpr auto operator+(index<a> const& l, index<b> const& r)
        -> index<a + b>
    {
        index<a + b> out;
        stdr::copy(r, stdr::copy(l, out.begin()).out);
        return out;
    }

    export namespace literals
    {
        template <istring str>
        inline consteval auto operator""_i() -> index<str> {
            return {};
        }
    }

    template <class T>
    inline constexpr istring to_istring = not defined(to_istring<T>);

    template <std::integral T>
    inline constexpr istring to_istring<T> = projection;

    template <istring str>
    inline constexpr istring to_istring<index<str>> = str;

    // namespace concepts
    // {
    //     template <class T>
    //     concept index = requires {
    //         typename std::decay_t<T>::_ttl_index_tag_type;
    //     };
    // }
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

    ttl::index ijp = i + j + p;
    assert(ijp[0] == 0);
    assert(ijp[1] == 0);
    assert(ijp[2] == 1);
    assert((std::same_as<decltype(ijp.projection_map()), std::index_sequence<2>>));

    ttl::index ipj = i + p + j;
    assert(ipj[0] == 0);
    assert(ipj[1] == 1);
    assert(ipj[2] == 0);
    assert((std::same_as<decltype(ipj.projection_map()), std::index_sequence<1>>));

    ttl::index pij = p + i + j;
    assert(pij[0] == 1);
    assert(pij[1] == 0);
    assert(pij[2] == 0);
    assert((std::same_as<decltype(pij.projection_map()), std::index_sequence<0>>));

    return true;
}

static_assert(test_index());
