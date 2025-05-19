#pragma once
#include <ttl/tree/sum.hpp>
import std;

namespace ttl::tree
{
	template <concepts::expression A, concepts::expression B>
	struct subtract : sum<A, B, std::minus{}> {
		using subtract::sum::sum;
	};

	template <concepts::expression A, concepts::expression B>
	inline constexpr auto operator-(A&& a, B&& b) -> subtract<A, B>
	{
		return subtract<A, B>(FWD(a), FWD(b));
	}
}
