module;

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>

module ttl:istring;
import :cstring;

namespace stdr = std::ranges;

namespace ttl
{
    inline constexpr char16_t projection[2] = { u'*', u'\0' };

    /// An index string.
    ///
    /// Extents the cstring with functions specific to tensor indices.
    template <std::size_t N>
    struct istring : cstring<N>
    {
        using istring::cstring::cstring;

        static constexpr auto projected = projection[0];

        constexpr auto rank() const -> std::size_t
        {
            return stdr::count_if(*this, [this](auto const& c) {
                return c != projected and this->count(c) == 1;
            });
        }

        constexpr auto outer() const -> istring
        {
            istring out;
            _unique(out.begin());
            return out;
        }

        constexpr auto inner() const -> istring
        {
            istring out;
            _contracted(_unique(out.begin()));
            return out;
        }

        constexpr auto all() const -> istring
        {
            istring out;
            _projected(_contracted(_unique(out.begin())));
            return out;
        }

        /// Generate the contracted indices.
        constexpr auto contracted() const -> istring
        {
            istring out;
            _contracted(out.begin());
            return out;
        }

        /// Check to see if this is a subset of b.
        template <std::size_t M>
        constexpr bool is_subset_of(istring<M> const& b) const
        {
            return std::ranges::all_of(*this, [&](char const c) {
                return b.count(c) != 0;
            });
        }

        /// Check to see if a is a permutation of b.
        template <std::size_t M>
        friend constexpr bool is_permutation(istring const& a, istring<M> const& b) {
            return a.is_subset_of(b) and b.is_subset_of(a);
        }

      private:
        /// Copy the unique indices, not including the projected character, into
        /// the output.
        constexpr auto _unique(auto *out) const -> auto* {
            return stdr::copy_if(*this, out, [this](auto const& c) {
                return c != projected and this->count(c) == 1;
            }).out;
        }

        /// Copy the contracted indices, not including the projected character,
        /// into the output.
        constexpr auto _contracted(auto *out) const -> auto* {
            // copy_if doesn't quite work for this because we want to
            // only insert each contracted variable once, and the
            // copy_if API doesn't give us access to the continuously
            // updating `out` in order to do that check
            auto const* i = out;
            for (auto const c : *this) {
                if (c != projected and this->count(c) == 2) {
                    if (stdr::count(i, out, c) == 0) {
                        *out++ = c;
                    }
                }
            }
            return out;
        }

        /// Copy the projected indices into the output.
        constexpr auto _projected(auto *out) const -> auto* {
            return stdr::copy_if(*this, out, [](auto const c) {
                return c == projected;
            }).out;
        }
    };

    export template <concepts::character T, std::size_t N>
    istring(T const (&str)[N]) -> istring<N>;
}

#undef DNDEBUG
static consteval bool test_istring()
{
    using ttl::istring;

    constexpr istring _ = "";
    assert(_.size() == 0);
    assert(_.rank() == 0);
    assert(_.outer() == _);
    assert(_.inner() == _);
    assert(_.all() == _);

    constexpr istring i = "i";
    assert(i.size() == 1);
    assert(i.rank() == 1);
    assert(i.outer() == i);
    assert(i.inner() == i);
    assert(i.all() == i);

    constexpr istring j = "j";
    constexpr istring ij = i + j;
    assert(ij.size() == 2);
    assert(ij.rank() == 2);
    assert(ij.outer() == ij);
    assert(ij.inner() == ij);
    assert(ij.all() == ij);

    constexpr istring iji = ij + i;
    assert(iji.outer() == j);
    assert(iji.inner() == j + i);
    assert(iji.all() == j + i);

    constexpr istring p = ttl::projection;
    assert(p.size() == 1);
    assert(p.rank() == 0);

    assert(p.outer() == _);
    assert(p.inner() == _);
    assert(p.all() == p);

    constexpr istring ip = i + p;
    assert(ip.size() == 2);
    assert(ip.rank() == 1);
    assert(ip.outer() == i);
    assert(ip.inner() == i);
    assert(ip.all() == ip);

    constexpr istring pi = p + i;
    assert(pi.size() == 2);
    assert(pi.rank() == 1);
    assert(pi.outer() == i);
    assert(pi.inner() == i);
    assert(pi.all() == ip);

    constexpr istring ppijip = p + p + iji + p;
    assert(ppijip.size() == 6);
    assert(ppijip.rank() == 1);
    assert(ppijip.outer() == j);
    assert(ppijip.inner() == j + i);
    assert(ppijip.all() == j + i + p + p + p);

    constexpr std::array map_p = index_of<ppijip, ppijip.projected>;
    assert(map_p.size() == 3);
    assert(map_p[0] == 0);
    assert(map_p[1] == 1);
    assert(map_p[2] == 5);

    constexpr std::array map_i = index_of<ppijip, 'i'>;
    assert(map_i.size() == 2);
    assert(map_i[0] == 2);
    assert(map_i[1] == 4);

    constexpr std::array map_j = index_of<ppijip, u'j'>;
    assert(map_j.size() == 1);
    assert(map_j[0] == 3);

    constexpr std::array map_k = index_of<ppijip, 'k'>;
    assert(map_k.size() == 0);

    return true;
}

static_assert(test_istring());
