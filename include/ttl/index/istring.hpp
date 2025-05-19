#pragma once

#include <ttl/index/cstring.hpp>
#include <cassert>

import std;

namespace ttl
{
	namespace stdr = std::ranges;
	
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

	template <concepts::character T, std::size_t N>
	istring(T const (&str)[N]) -> istring<N>;
}
