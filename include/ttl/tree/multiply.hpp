#pragma once

#include <ttl/tree/product.hpp>
import std;

namespace ttl::tree
{
	template <concepts::expression A, concepts::expression B>
	struct multiply : product<A, B, std::multiplies {}, std::plus {}> {
		using multiply::product::product;
	};

	template <concepts::expression A, concepts::expression B>
	constexpr auto operator*(A&& a, B&& b) -> multiply<A, B>
	{
		return multiply<A, B>(FWD(a), FWD(b));
	}
}
