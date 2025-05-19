#include <ttl/tensor_traits.hpp>

using namespace ttl::concepts;

namespace
{
	struct S {};
}

template <>
struct ttl::tensor_traits<S> {
	static constexpr auto extents(S&&) -> std::extents<std::size_t, 1> {
		return {};
	}

	static constexpr auto evaluate(S&&, int) -> int {
		return 0;
	}

	static constexpr std::integral_constant<std::size_t, 1> rank;
};

static_assert(has_extents_trait<S>);
static_assert(has_evaluate_trait<S, int>);
static_assert(has_rank_trait<S>);
