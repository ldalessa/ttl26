#undef DNDEBUG

#include <ttl/std.hpp>
#include <ttl/tree/bind.hpp>
#include <ttl/tree/sum.hpp>

static constexpr bool test_scalar()
{
    auto b = ttl::bind(1) + ttl::bind(1);
    return true;
}

static_assert(test_scalar());
