#pragma once
#include <ttl/tree/sum.hpp>

import std;

namespace ttl::tree
{
	template <concepts::expression A, concepts::expression B>
	struct add : sum<A, B, std::plus{}> {
		using add::sum::sum;
	};

	
	template <concepts::expression A, concepts::expression B>
	inline constexpr auto operator+(A&& a, B&& b) -> add<A, B>
	{
		return add<A, B>(FWD(a), FWD(b));
	}
}
