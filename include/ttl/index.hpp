#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <format>
#include <iterator>
#include <ranges>
#include <type_traits>

namespace ttl
{
    template <std::size_t N = 1>
    struct index
    {
        static_assert(N != 0);
        using _char_t = char;
        using _tensor_index_tag_t = void;

        _char_t _indices[N]{};

        constexpr index() = default;

        constexpr index(_char_t const (&str)[N]) {
            std::ranges::copy(str, _indices);
        }

        template <std::size_t M>
        constexpr auto operator==(this index const& a, index<M> const& b) -> bool {
            return std::ranges::equal(a, b);
        }

        /// Concatenate indices ("ii" + "j" -> "iij").
        template <std::size_t M>
        constexpr auto operator+(this index const& a, index<M> const& b) -> index<N + M - 1>
        {
            index<N + M - 1> out{};             // only need one null terminator
            std::ranges::copy(b, std::ranges::copy(a, out.begin()).out);
            return out;
        }

        template <std::size_t M>
        constexpr auto operator^(this index const& a, index<M> const& b) -> index<N + M - 1> {
            return (a + b).exported();      // probably cheaper to do this directly
        }

        template <std::size_t M>
        constexpr auto operator&(this index const& a, index<M> const& b) -> index<N + M - 1> {
            return (a + b).contracted();      // probably cheaper to do this directly
        }

        /// Produce the uncontracted indices in the index ("iij" -> "j").
        constexpr auto exported() const -> index
        {
            index out{};
            std::ranges::copy_if(*this, out.begin(), [&](_char_t const c) {
                return count(c) == 1;
            });
            return out;
        }

        /// Produce the contracted indices in the index ("iij" -> "i");
        constexpr auto contracted() const -> index
        {
            index out{};
            std::ranges::copy_if(*this, out.begin(), [&](_char_t const c) {
                return count(c) == 2 and out.count(c) == 0;
            });
            return out;
        };

        /// Check to see if a is a subset of b.
        ///
        /// @precondition a should not should not have any contracted indices
        template <std::size_t M>
        constexpr bool is_subset_of(this index const& a, index<M> const& b)
        {
            return std::ranges::all_of(a, [&](_char_t const c) {
                assert(a.count(c) == 1); // shouldn't have any contracted indices
                return b.count(c) != 0;
            });
        }

        /// Check to see if a is a permutation of b.
        ///
        /// @precondition neither a nor b should have contracted indices
        template <std::size_t M>
        constexpr bool is_permutation_of(this index const& a, index<M> const& b) {
            return a.is_subset_of(b) and b.is_subset_of(a);
        }

        /// Compute the size of the index.
        constexpr auto size() const -> std::size_t {
            return __builtin_strlen(_indices);
        }

        /// Compute the rank of the index.
        constexpr auto rank() const -> std::size_t {
            return std::ranges::count_if(*this, [&](_char_t const c) {
                return count(c) == 1;
            });
        }

        constexpr auto operator[](std::size_t const i) const -> _char_t {
            assert(i < size());
            return _indices[i];
        }

        constexpr auto count(_char_t const c) const -> std::size_t {
            return std::ranges::count(*this, c);
        }

        constexpr auto index_of(_char_t const c) const {
            return std::ranges::distance(begin(), std::ranges::find(*this, c));
        }

        template <std::size_t R, std::size_t M>
        constexpr auto index_of(index<M> const& b) const -> std::array<int, R> {
            assert(R = b.rank());
            std::array<int, M> out{};
            for (int i = 0; _char_t const c : b) {
                out[i++] = index_of(c);
            }
            return out;
        }

        constexpr auto end(this auto&) {
            struct _sentinel{};
            return _sentinel{};
        }

        constexpr auto begin(this auto& self) -> std::forward_iterator auto
        {
            using _sentinel = decltype(self.end());

            struct _iterator
            {
                using value_type = std::remove_reference_t<decltype(self._indices[0])>;
                using difference_type = std::ptrdiff_t;

                value_type* i;

                constexpr bool operator==(_iterator const&) const = default;
                constexpr bool operator==(_sentinel) const {
                    return *i == '\0';
                }

                constexpr auto operator*() const -> value_type& {
                    return *i;
                }

                constexpr auto operator++() -> _iterator& {
                    ++i; return *this;
                }

                constexpr auto operator++(int) -> _iterator; // needed for forward_iterator
            };

            static_assert(std::forward_iterator<_iterator>);
            static_assert(std::sentinel_for<_sentinel, _iterator>);

            return _iterator(self._indices);
        }

        constexpr auto format(auto& ctx) const -> decltype(std::format_to(ctx.out(), "{}", _indices)) {
            return std::format_to(ctx.out(), "{}", _indices);
        }
    };

    namespace concepts
    {
        template <class T>
        concept tensor_index = requires {
            typename std::remove_reference_t<T>::_tensor_index_tag_t;
        };
    }
}
