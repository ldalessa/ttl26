module;

#include <array>
#include <utility>

module ttl:imap;
import :concepts;
import :istring;

namespace ttl
{
    /// Create an index map from `from` to `to.
    ///
    /// This means if `from[i] -> to[j]` then `map[j] == i`.
    template <istring from, istring to>
    inline constexpr concepts::index_sequence auto imap = []
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

        return []<std::size_t... i>(std::index_sequence<i...>) {
            return std::index_sequence<map[i]...>();
        }(std::make_index_sequence<size>());
    }();
}
