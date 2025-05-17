module;
#include <cassert>
export module ttl:imap;
export import :istring;
import std;

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
	static constexpr istring all = "i*";
	static constexpr auto map = imap<all, all>;
	// print<map> _;
	assert((std::same_as<decltype(auto(map)), std::integer_sequence<unsigned long,0,1>>));

	static constexpr istring a = "ij";
	static constexpr istring b = "ji";

	static constexpr auto ab = imap<a, b>;
	static constexpr auto ba = imap<b, a>;

	assert((std::same_as<decltype(auto(ab)), std::integer_sequence<unsigned long,1,0>>));	
	assert((std::same_as<decltype(auto(ba)), std::integer_sequence<unsigned long,1,0>>));

	static constexpr istring x = "ijk";
	static constexpr istring y = "jki";

	static constexpr auto xy = imap<x, y>;
	static constexpr auto yx = imap<y, x>;
	
	assert((std::same_as<decltype(auto(xy)), std::integer_sequence<unsigned long,1,2,0>>));
	assert((std::same_as<decltype(auto(yx)), std::integer_sequence<unsigned long,2,0,1>>));
	
	return true;
}

static_assert(test_imap());
