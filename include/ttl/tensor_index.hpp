#pragma once

#include <ttl/ARROW.hpp>
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
    struct tensor_index
    {
        static_assert(N != 0);
        using _char_t = char;
        using _tensor_index_tag_t = void;

        _char_t _indices[N]{};

        constexpr tensor_index() = default;

        constexpr tensor_index(_char_t const (&str)[N]) {
            std::ranges::copy(str, _indices);
        }

        template <std::size_t M>
        friend constexpr bool operator==(tensor_index const& a, tensor_index<M> const& b) {
            return std::ranges::equal(a, b);
        }

        /// Concatenate indices ("ii" + "j" -> "iij").
        template <std::size_t M>
        friend constexpr auto operator+(tensor_index const& a, tensor_index<M> const& b) -> tensor_index<N + M - 1>
        {
            tensor_index<N + M - 1> out{};             // only need one null terminator
            std::ranges::copy(b, std::ranges::copy(a, out.begin()).out);
            return out;
        }

        /// Produce the uncontracted indices in the index ("iij" -> "j").
        constexpr auto exported() const -> tensor_index
        {
            tensor_index out{};
            std::ranges::copy_if(*this, out.begin(), [&](_char_t const c) {
                return count(c) == 1;
            });
            return out;
        }

        /// Produce the contracted indices in the tensor_index ("iij" -> "i");
        constexpr auto contracted() const -> tensor_index
        {
            tensor_index out{};
            std::ranges::copy_if(*this, out.begin(), [&](_char_t const c) {
                return count(c) == 2 and out.count(c) == 0;
            });
            return out;
        };

        /// Check to see if a is a subset of b.
        ///
        /// @precondition a should not should not have any contracted indices
        template <std::size_t M>
        constexpr bool is_subset_of(tensor_index<M> const& b) const
        {
            return std::ranges::all_of(*this, [&](_char_t const c) {
                switch (b.count(c)) {
                  case 1: return true;
                }
                assert(false && "b should not contain repeated indices");
            });
        }

        /// Check to see if a is a permutation of b.
        ///
        /// @precondition neither a nor b should have contracted indices
        template <std::size_t M>
        constexpr bool is_permutation_of(tensor_index<M> const& b) const {
            return this->is_subset_of(b) and b.is_subset_of(*this);
        }

        /// Compute the size of the tensor_index.
        constexpr auto size() const -> std::size_t {
            return __builtin_strlen(_indices);
        }

        /// Compute the rank of the tensor_index.
        constexpr auto rank() const -> std::size_t {
            return std::ranges::count_if(*this, [&](_char_t const c) {
                return count(c) == 1;
            });
        }

        constexpr auto data() const -> char const* {
            return _indices;
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
        friend constexpr auto index_of(tensor_index const& self, tensor_index<M> const& b) -> std::array<int, R> {
            assert(R == b.rank());
            std::array<int, R> out{};
            for (int i = 0; _char_t const c : b) {
                out[i++] = self.index_of(c);
            }
            return out;
        }

        struct _sentinel{};
        static constexpr auto _end(auto&&) -> _sentinel {
            return {};
        }

        constexpr auto end() const TTL_ARROW( _end(*this) );
        constexpr auto end()       TTL_ARROW( _end(*this) );

        static constexpr auto _begin(auto& indices) -> std::forward_iterator auto
        {
            struct _iterator
            {
                using value_type = std::remove_reference_t<decltype(indices[0])>;
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

            return _iterator(indices);
        }
        constexpr auto begin() const TTL_ARROW( _begin(_indices) );
        constexpr auto begin()       TTL_ARROW( _begin(_indices) );


        constexpr auto format(auto& ctx) const -> decltype(std::format_to(ctx.out(), "{}", _indices)) {
            return std::format_to(ctx.out(), "{}", _indices);
        }
    };
}
