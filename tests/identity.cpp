#undef DNDEBUG

#include <ttl/index/index.hpp>
#include <ttl/tree/bind.hpp>
#include <ttl/tree/identity.hpp>

using namespace ttl;
using namespace ttl::tree;

static constexpr bool test_identity_scalar()
{
	auto a = bind(1);
	auto b = +a;
	assert(b == 1);
	
	return true;
}

static_assert(test_identity_scalar());

static constexpr bool test_identity_vector()
{
	static constexpr ttl::index<"i"> i;
	
	int const x[] = {1, 2};
	auto a = bind(x, i);
	auto b = +a;
	assert(b[0] == 1);
	assert(b[1] == 2);
	return true;
}

static_assert(test_identity_vector());

static constexpr bool test_identity_matrix()
{
	static constexpr ttl::index<"i"> i;
	static constexpr ttl::index<"j"> j;
	
	int const x[2][1] = {{1}, {2}};
	auto a = bind(x, i, j);
	auto b = +a;
	assert((b[0,0] == 1));
	assert((b[1,0] == 2));
	return true;
}

static_assert(test_identity_matrix());
