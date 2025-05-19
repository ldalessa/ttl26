#undef DNDEBUG
#include <ttl/tensor/rank.hpp>

using namespace ttl;


static_assert(rank<int> == 0);
static_assert(rank<int[1]> == 1);
static_assert(rank<int[1][1]> == 2);

static_assert(rank<int&> == 0);
static_assert(rank<int(&)[1]> == 1);
static_assert(rank<int(&)[1][1]> == 2);

static_assert(rank<int&&> == 0);
static_assert(rank<int(&&)[1]> == 1);
static_assert(rank<int(&&)[1][1]> == 2);

static_assert(rank<int const> == 0);
static_assert(rank<int const[1]> == 1);
static_assert(rank<int const[1][1]> == 2);

static_assert(rank<int const&> == 0);
static_assert(rank<int const(&)[1]> == 1);
static_assert(rank<int const(&)[1][1]> == 2);

static_assert(rank<int const&&> == 0);
static_assert(rank<int const(&&)[1]> == 1);
static_assert(rank<int const(&&)[1][1]> == 2);

static_assert(rank<std::span<int, 1>> == 1);
static_assert(rank<std::span<int[1], 1>> == 2);
static_assert(rank<std::span<int[1][1], 1>> == 3);

static_assert(rank<std::span<int const, 1>> == 1);
static_assert(rank<std::span<int const[1], 1>> == 2);
static_assert(rank<std::span<int const[1][1], 1>> == 3);

static_assert(rank<std::array<int, 1>> == 1);
static_assert(rank<std::array<int[1], 1>> == 2);
static_assert(rank<std::array<int[1][1], 1>> == 3);

static_assert(rank<std::array<int const, 1>> == 1);
static_assert(rank<std::array<int const[1], 1>> == 2);
static_assert(rank<std::array<int const[1][1], 1>> == 3);

static_assert(rank<std::vector<int>> == 1);
static_assert(rank<std::vector<int[1]>> == 2);
static_assert(rank<std::vector<int[1][1]>> == 3);

static_assert(rank<std::mdspan<int, std::extents<int>>> == 0);
static_assert(rank<std::mdspan<int, std::extents<int, 1>>> == 1);
static_assert(rank<std::mdspan<int, std::extents<int, 1, 1>>> == 2);
static_assert(rank<std::mdspan<int, std::extents<int, 1, std::dynamic_extent, 1>>> == 3);

static_assert(rank<std::mdspan<int const, std::extents<int>>> == 0);
static_assert(rank<std::mdspan<int const, std::extents<int, 1>>> == 1);
static_assert(rank<std::mdspan<int const, std::extents<int, 1, 1>>> == 2);
static_assert(rank<std::mdspan<int const, std::extents<int, 1, std::dynamic_extent, 1>>> == 3);

static_assert(rank<std::mdspan<std::array<int, 3>, std::extents<int>>> == 1);
static_assert(rank<std::mdspan<std::array<int, 3>, std::extents<int, 1>>> == 2);
static_assert(rank<std::mdspan<std::array<int, 3>, std::extents<int, 1, 1>>> == 3);
static_assert(rank<std::mdspan<std::array<int, 3>, std::extents<int, 1, std::dynamic_extent, 1>>> == 4);
