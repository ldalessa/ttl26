#undef DNDEBUG
#include <ttl/tensor/tensor.hpp>

using namespace ttl;
using namespace ttl::concepts;

static_assert(tensor<float>);

static_assert(tensor<int>);
static_assert(tensor<int&>);
static_assert(tensor<int&&>);

static_assert(tensor<int const>);
static_assert(tensor<int const&>);
static_assert(tensor<int const&&>);

static_assert(tensor<int[3]>);
static_assert(tensor<int(&)[3]>);
static_assert(tensor<int(&&)[3]>);

static_assert(tensor<int const[3]>);
static_assert(tensor<int const(&)[3]>);
static_assert(tensor<int const(&&)[3]>);

static_assert(tensor<int[3][3]>);
static_assert(tensor<int(&)[3][3]>);
static_assert(tensor<int(&&)[3][3]>);

static_assert(tensor<int const[3][3]>);
static_assert(tensor<int const(&)[3][3]>);
static_assert(tensor<int const(&&)[3][3]>);

static_assert(tensor<std::span<int, 3>>);
static_assert(tensor<std::span<int[3], 3>>);
static_assert(tensor<std::span<int const, 3>>);
static_assert(tensor<std::span<int const[3], 3>>);

static_assert(tensor<std::array<int, 3>>);
static_assert(tensor<std::array<int[3], 3>>);
static_assert(tensor<std::array<int const, 3>>);
static_assert(tensor<std::array<int const[3], 3>>);

static_assert(tensor<std::mdspan<int, std::extents<std::size_t>>>);
static_assert(tensor<std::mdspan<int, std::extents<std::size_t, 3>>>);
static_assert(tensor<std::mdspan<int, std::extents<std::size_t, 3, std::dynamic_extent>>>);

static_assert(tensor<std::mdspan<int const, std::extents<std::size_t>>>);
static_assert(tensor<std::mdspan<int const, std::extents<std::size_t, 3>>>);
static_assert(tensor<std::mdspan<int const, std::extents<std::size_t, 3, std::dynamic_extent>>>);

static_assert(tensor<std::vector<int>>);
static_assert(tensor<std::vector<std::vector<int>>>);

static_assert(scalar<int>);
static_assert(scalar<int&>);
static_assert(scalar<int&&>);

static_assert(scalar<int const>);
static_assert(scalar<int const&>);
static_assert(scalar<int const&&>);

static_assert(scalar<float>);
static_assert(scalar<float&>);
static_assert(scalar<float&&>);

static_assert(scalar<float const>);
static_assert(scalar<float const&>);
static_assert(scalar<float const&&>);

static_assert(not expression<int[3]>);
static_assert(not expression<int(&)[3]>);
static_assert(not expression<int(&&)[3]>);

static_assert(not expression<int[3][3]>);
static_assert(not expression<int(&)[3][3]>);
static_assert(not expression<int(&&)[3][3]>);
