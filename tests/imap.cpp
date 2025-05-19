#undef DNDEBUG

#include <ttl/index/imap.hpp>

using namespace ttl;

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
