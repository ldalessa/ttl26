module;

#include <array>
#include <cassert>
#include <concepts>
#include <utility>

module ttl:imap;
import :istring;

namespace ttl
{
    /// Create an index map from `from` to `to.
    ///
    /// This means if `from[i] -> to[j]` then `map[j] == i`.
    template <istring from, istring to>
    inline constexpr std::integer_sequence imap = []
    {
        static_assert(to.is_subset_of(from));

        static constexpr auto size = to.size();

        static constexpr auto map = [] {
            std::array<std::size_t, size> out;
            int p = 0;
            int i = 0;
            for (auto const c : to) {
                if (c == to.projected) {
                    out[i++] = from.index_of_1_nth(c, p++);
                }
                else {
                    out[i++] = from.index_of_1(c);
                }
            }
            return out;
        }();

        return to_sequence<map>;
    }();
}

using namespace ttl;

#undef DNDEBUG

static constexpr bool test_imap()
{
    constexpr istring all = "i*";
    constexpr auto map = imap<all, all>;
    // print<map> _;
    // assert((std::same_as<decltype(auto(map)), std::integer_sequence<unsigned long,0,1>>));

    return true;
}

static_assert(test_imap());
