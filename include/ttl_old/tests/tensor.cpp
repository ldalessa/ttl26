#undef DNDEBUG

#include <ttl/ttl.hpp>

static_assert(ttl::scalar<int>);
static_assert(ttl::scalar<int const>);
static_assert(ttl::expression<int>);

static_assert(ttl::scalar<float>);
static_assert(ttl::scalar<float const>);
static_assert(ttl::expression<float>);

static_assert(ttl::tensor_of_rank<std::span<int>, 1>);
static_assert(ttl::tensor_of_rank<std::span<int const>, 1>);
static_assert(ttl::tensor_of_rank<std::span<std::span<int>>, 2>);
static_assert(ttl::tensor_of_rank<std::span<std::span<int const>>, 2>);
static_assert(not ttl::expression<std::span<int>>);

static_assert(ttl::tensor_of_rank<int[1], 1>);
static_assert(ttl::tensor_of_rank<int const[1], 1>);
static_assert(ttl::tensor_of_rank<int[1][1], 2>);
static_assert(ttl::tensor_of_rank<int const[1][1], 2>);

static_assert(ttl::tensor_of_rank<std::array<int, 1>, 1>);
static_assert(ttl::tensor_of_rank<std::array<int const, 1>, 1>);
static_assert(ttl::tensor_of_rank<std::array<int[1], 1>, 2>);
static_assert(ttl::tensor_of_rank<std::array<int const[1], 1>, 2>);

static_assert(ttl::tensor_of_rank<std::vector<int>, 1>);
static_assert(ttl::tensor_of_rank<std::vector<std::vector<int>>, 2>);

static_assert(ttl::tensor_of_rank<std::mdspan<int, std::extents<std::size_t>>, 0>);
static_assert(ttl::tensor_of_rank<std::mdspan<int, std::extents<std::size_t, 1>>, 1>);
static_assert(ttl::tensor_of_rank<std::mdspan<int, std::extents<std::size_t, std::dynamic_extent>>, 1>);
static_assert(ttl::tensor_of_rank<std::mdspan<int, std::extents<std::size_t, std::dynamic_extent, 1>>, 2>);

int main()
{
    return 0;
}
