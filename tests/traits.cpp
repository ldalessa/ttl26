#include <ttl/std.hpp>
#include <ttl/tensor.hpp>

using namespace ttl::concepts;

template <class T, std::size_t N>
static constexpr bool _verify_tensor_traits = []
 {
     static_assert(tensor_of_rank<T, N>);
     static_assert(tensor_of_rank<T const, N>);
     static_assert(tensor_of_rank<T&, N>);
     static_assert(tensor_of_rank<T&&, N>);
     static_assert(tensor_of_rank<T const&, N>);
     static_assert(tensor_of_rank<T const&&, N>);
     return true;
 }();

template <class T>
static constexpr bool _verify_expression_traits = []
 {
     static_assert(expression<T>);
     static_assert(expression<T const>);
     static_assert(expression<T&>);
     static_assert(expression<T&&>);
     static_assert(expression<T const&>);
     static_assert(expression<T const&&>);
     return true;
 }();

static_assert(_verify_tensor_traits<int, 0>);
static_assert(_verify_expression_traits<int>);
static_assert(_verify_tensor_traits<float, 0>);
static_assert(_verify_expression_traits<float>);

static_assert(_verify_tensor_traits<std::vector<int>, 1>);
static_assert(_verify_tensor_traits<std::vector<int const>, 1>);
static_assert(_verify_tensor_traits<std::vector<std::vector<int>>, 2>);

static_assert(_verify_tensor_traits<int[1], 1>);
static_assert(_verify_tensor_traits<int[1][1], 2>);
static_assert(_verify_tensor_traits<int const[1][1][1], 3>);

static_assert(_verify_tensor_traits<std::array<int, 1>, 1>);
static_assert(_verify_tensor_traits<std::array<int const, 1>, 1>);
static_assert(_verify_tensor_traits<std::array<std::array<int, 1>, 1>, 2>);

static_assert(_verify_tensor_traits<std::span<int, 1>, 1>);
static_assert(_verify_tensor_traits<std::span<int const, 1>, 1>);
static_assert(_verify_tensor_traits<std::span<std::array<int, 1>, 1>, 2>);
static_assert(_verify_tensor_traits<std::span<int>, 1>);

static_assert(_verify_tensor_traits<std::mdspan<int, std::extents<int, 1>>, 1>);
static_assert(_verify_tensor_traits<std::mdspan<int const, std::extents<int, 1>>, 1>);
static_assert(_verify_tensor_traits<std::mdspan<int, std::extents<int, 1, std::dynamic_extent>>, 2>);
static_assert(_verify_tensor_traits<std::mdspan<int, std::extents<int, 1, std::dynamic_extent, 3>>, 3>);
