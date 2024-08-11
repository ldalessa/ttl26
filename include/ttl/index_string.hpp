#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <ranges>
#include <utility>

namespace ttl
{
    /// This character is used to represent a projected index in an
    /// index_string. A projected index is one where the user has specified an
    /// integer. For example A(i,2) is projecting the 3rd column of A, and the
    /// corresponding index_string will be "i*".
    static constexpr char projected_index = '*';

    template <std::size_t N = 1>
    struct index_string {
        static_assert(0 < N); // always store at least a '\0'

        char _data[N] {};

        consteval index_string() = default;

        /// Construct an index string from a c-string.
        consteval index_string(char const (&str)[N])
        {
            // std::strcpy is not constexpr, so we use our cstring sentinel and
            // use copy
            auto const in = std::ranges::subrange(str, _cstring_sentinel {});
            std::ranges::copy(in, _data);
        }

        /// Get a specific index from the index_string.
        constexpr auto operator[](std::size_t i) const -> char
        {
            assert(i < size());
            return _data[i];
        }

        /// The size of the index is the length of the string.
        ///
        /// @note(performance)
        ///
        /// I have chosen to store the string as a null-terminated string
        /// because then compiler and debugger output prints the string rather
        /// than an array of characters. I have chsen not to store the size
        /// because it is redundant and it needs to be kept consistent and makes
        /// various operations more complicated.
        ///
        /// Strings are short and this is all done in constexpr, so I'm
        /// relatively happy with this decision. It's something that could be
        /// re-evaluated if compilation times become extreme.
        constexpr auto size() const -> std::size_t
        {
            return __builtin_strlen(_data);
        }

        /// The rank of the index is the number of outer indices.
        ///
        /// Outer indices are those that are neither contracted nor
        /// projected. Contracted indices are those that are not projected and
        /// appear twice in _data. Projected indices are those that are the
        /// projected_index (currently `'*'`).
        constexpr auto rank() const -> std::size_t
        {
            return std::ranges::count_if(*this, [&](char const c) {
                return c != projected_index and count(c) == 1;
            });
        }

        /// Use a raw pointer as the iterator type.
        constexpr auto begin(this auto& self)
        {
            return self._data;
        }

        /// A cstring sentinel just needs to check its partner for  '\0'.
        struct _cstring_sentinel {
            constexpr bool operator==(char const* str) const
            {
                return *str == '\0';
            }
        };

        /// Create an end sentinel.
        ///
        /// Using a sentinel means that `end()` can't be used for some normal
        /// operations, but its advantage is that we don't ever need to do
        /// strlen to use the index_string as a cstring range.
        static constexpr auto end() -> _cstring_sentinel
        {
            return {};
        }

        /// Test if two index strings are equal.
        template <std::size_t M>
        constexpr bool operator==(index_string<M> const& b) const
        {
            return __builtin_strcmp(_data, b._data) == 0;
        }

        /// Concatenate two index_strings.
        ///
        /// The resulting string is N + M - 1 because we need at most 1 '\0' and
        /// N + M has two.
        template <std::size_t M>
        constexpr auto operator+(index_string<M> const& b) const -> index_string<N + M - 1>
        {
            index_string<N + M - 1> out;
            auto [_, i] = std::ranges::copy(*this, out._data);
            std::ranges::copy(b, i);
            return out;
        }

        /// This set of query functions creates new index_string<N> that contain
        /// the requested subset of indices. We do these explicitly because we
        /// know they can never get any larger than N, so this prevents N
        /// explosion during concatenation.  @{

        /// Returns the outer indices (see `_outer`).
        constexpr auto outer() const -> index_string
        {
            index_string out;
            _outer(out._data);
            return out;
        }

        /// Returns the contracted indices (see `_contracted`).
        constexpr auto contracted() const -> index_string
        {
            index_string out;
            _contracted(out, out._data);
            return out;
        }

        /// Returns the projected indices (see `_projected`).
        constexpr auto projected() const -> index_string
        {
            index_string out;
            _projected(out._data);
            return out;
        }

        /// Returns outer() + contracted().
        constexpr auto inner() const -> index_string
        {
            index_string out;
            _contracted(out, _outer(out._data));
            return out;
        }

        /// Returns outer() + contracted() + projected().
        constexpr auto all() const -> index_string
        {
            index_string out;
            _projected(_contracted(out, _outer(out._data)));
            return out;
        }

        /// @}

        /// Find the index of `c`.
        ///
        /// This returns the index of the first instance of `c`. If `c` is
        /// contracted or projected this may not be what the caller wants.
        constexpr auto index_of(char const c) const -> std::size_t
        {
            return std::ranges::distance(begin(), std::ranges::find(*this, c));
        }

        /// Find the index of `c`, with special handling for projected indices.
        ///
        /// If `c` is not the projected index '*' then this is the same as
        /// index_of(c). If `c` is the projected index then this finds the pth
        /// indstance of it and returns that index, and updates `p` by one.
        ///
        /// The point is that this can be used transparently in a loop that is
        /// tracking projected offsets.
        ///
        /// @precondition If `c == projected_index` then the number of
        /// instances of `'*'` must be at least `p`.
        ///
        /// @param[in] c The index we're looking for.
        /// @param[in/out] p The current count of projected indices.
        ///
        /// @returns The found index, and updates `p` if `c` was the projected
        /// index.
        constexpr auto index_of(char const c, int& p) const -> std::size_t
        {
            if (c != projected_index) {
                return index_of(c);
            } else {
                int n = p++;
                return std::ranges::distance(begin(), std::ranges::find_if(*this, [&](const char d) {
                    return d == c and 0 == n--;
                }));
            }
            return std::ranges::distance(begin(), std::ranges::find(*this, c));
        }

        /// Count the number of incstaces of `c`.
        constexpr auto count(char const c) const -> std::size_t
        {
            return std::ranges::count(*this, c);
        }

        /// Check to see if a is a subset of b.
        template <std::size_t M>
        constexpr bool is_subset_of(index_string<M> const& b) const
        {
            return std::ranges::all_of(*this, [&](char const c) {
                return b.count(c) != 0;
            });
        }

        /// Find the two offsets for a contracted character.
        ///
        /// @precondition `c` is a contracted index
        /// @param c The contracted index to search for.
        ///
        /// @returns The pair of indices as an array.
        constexpr auto find_offsets(char const c) const -> std::array<int, 2>
        {
            assert(c != projected_index);
            std::array<int, 2> out;
            int i = 0; // index of d
            int j = 0; // index into out
            for (auto const d : *this) {
                if (c == d)
                    out[j++] = i;
                i += 1;
            }
            assert(out[0] != out[1]);
            return out;
        }

    private:
        /// Append my outer indices to the `out` iterator.
        ///
        /// Outer indices are those that are not '*' and only appear once. These
        /// are indices that are neither projected nor contracted.
        ///
        /// @param out The output iterator.
        ///
        /// @returns The updated out iterator.
        constexpr auto _outer(char* out) const -> char*
        {
            return std::ranges::copy_if(*this, out, [&](char const c) {
                return c != projected_index and count(c) == 1;
            }).out;
        }

        /// Append my contracted indices to the `out` iterator.
        ///
        /// Contracted indices are those that are not '*' and appear twice. Each
        /// contracted index must be added only once to the output iterator. In
        /// order to check this, the output `index_string` corresponding to the
        /// `out` iterator is also provided.
        ///
        /// @param str The output string (used to search for duplicates).
        /// @param out The output iterator (points into `str`).
        ///
        /// @returns The updated `out` iterator.
        constexpr auto _contracted(index_string const& str, char* out) const -> char*
        {
            return std::ranges::copy_if(*this, out, [&](char const c) {
                return c != projected_index and count(c) == 2 and str.count(c) == 0;
            }).out;
        }

        /// Append the projected indices.
        ///
        /// Projected indices are those corresponding to '*'. Every instance of
        /// '*' is coppied to the output.
        ///
        /// @param out The output iterator.
        ///
        /// @returns The updated `out` iterator.
        constexpr auto _projected(char* out) const -> char*
        {
            return std::ranges::copy_if(*this, out, [&](char const c) {
                return c == projected_index;
            }).out;
        }
    };

    /// Check to see if a is a permutation of b.
    template <std::size_t N, std::size_t M>
    static constexpr bool is_permutation(index_string<N> const& a, index_string<M> const& b)
    {
        return a.is_subset_of(b) and b.is_subset_of(a);
    }

    /// Turn a constexpr array into an index sequence.
    template <std::array a>
    inline constexpr auto index_sequence_from_array = []<std::size_t... i>(std::index_sequence<i...>) {
        return std::index_sequence<a[i]...>();
    }(std::make_index_sequence<a.size()>());

    /// Create an index map from `from` to `to.
    ///
    /// This means if `from[i] -> to[j]` then `map[j] == i`.
    template <index_string from, index_string to>
    inline constexpr auto index_map = [] {
        static_assert(to.is_subset_of(from));

        static constexpr auto map = [] {
            std::array<std::size_t, to.size()> out;
            int p = 0;
            int i = 0;
            for (char const c : to) {
                out[i++] = from.index_of(c, p);
            }
            return out;
        }();

        return index_sequence_from_array<map>;
    }();
}
