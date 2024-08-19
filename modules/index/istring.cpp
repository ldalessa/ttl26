module;

#include <algorithm>
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

        constexpr auto rank() const -> std::size_t {
            return stdr::count_if(*this, [this](auto const& c) {
                return this->count(c) == 1;
            });
        }

        constexpr auto n_projected() const -> std::size_t {
            return this->count(projected);
        }

        constexpr auto outer() const -> istring {
            istring out;
            _unique(out.begin());
            return out;
        }

        constexpr auto inner() const -> istring {
            istring out;
            _contracted(_unique(out.begin()));
            return out;
        }

        constexpr auto all() const -> istring {
            istring out;
            _projected(_contracted(_unique(out.begin())));
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
        constexpr auto _unique(auto *out) const -> auto* {
            return stdr::copy_if(*this, out, [this](auto const& c) {
                return this->count(c) == 1;
            }).out;
        }

        constexpr auto _contracted(auto *out) const -> auto* {
            // copy_if doesn't quite work for this because we want to
            // only insert each contracted variable once, and the
            // copy_if API doesn't give us access to the continuously
            // updating `out` in order to do that check
            auto const* i = out;
            for (auto const c : *this) {
                if (this->count(c) == 2) {
                    if (stdr::count(i, out, c) == 0) {
                        *out++ = c;
                    }
                }
            }
            return out;
        }

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

    constexpr istring i = "i";
    assert(i.outer() == i);
    assert(i.inner() == i);
    assert(i.all() == i);

    constexpr istring j = "j";
    constexpr auto ij = i + j;
    assert(ij.outer() == ij);
    assert(ij.inner() == ij);
    assert(ij.all() == ij);

    constexpr auto iji = ij + i;
    assert(iji.outer() == j);
    assert(iji.inner() == j + i);
    assert(iji.all() == j + i);

    constexpr istring p = ttl::projection;
    constexpr auto ppijip = p + p + iji + p;
    assert(ppijip.outer() == j);
    assert(ppijip.inner() == j + i);
    assert(ppijip.all() == j + i + p + p + p);

    return true;
}

static_assert(test_istring());
