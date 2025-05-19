#undef DNDEBUG

#include <ttl/tree/multiply.hpp>
#include <ttl/tree/product.hpp>

using namespace ttl;
using namespace ttl::tree;

static constexpr bool test_scalar()
{
	return true;
}

static_assert(test_scalar());
