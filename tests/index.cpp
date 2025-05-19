#undef NDEBUG

#include <ttl/index/index.hpp>

static consteval bool test_index() {
	using namespace ttl::literals;
	ttl::istring n = "n";
	ttl::istring m = "m";
	(void)(n + m);

	ttl::index<"i"> i;
	auto j = "j"_i;
	ttl::index p(1);

	ttl::index ijp = i + j + p;
	assert(ijp[0] == 0);
	assert(ijp[1] == 0);
	assert(ijp[2] == 1);
	assert((std::same_as<decltype(ijp.projection_map()), std::index_sequence<2>>));

	ttl::index ipj = i + p + j;
	assert(ipj[0] == 0);
	assert(ipj[1] == 1);
	assert(ipj[2] == 0);
	assert((std::same_as<decltype(ipj.projection_map()), std::index_sequence<1>>));

	ttl::index pij = p + i + j;
	assert(pij[0] == 1);
	assert(pij[1] == 0);
	assert(pij[2] == 0);
	assert((std::same_as<decltype(pij.projection_map()), std::index_sequence<0>>));

	return true;
}

static_assert(test_index());
