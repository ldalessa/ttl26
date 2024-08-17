module;

#include <array>
#include <concepts>
#include <cstdint>
#include <mdspan>
#include <ranges>
#include <utility>
#include <vector>

module ttl:tensor;
import :extents;
import :evaluate;
import :outer;
import :rank;

namespace ttl
{
    template <class T>
    concept tensor = has_extents<T> and has_evaluate_n<T, rank<T>>;

    template <class T>
    concept expression = tensor<T> and has_outer<T>;

    template <class T>
    concept scalar = expression<T> and rank<T> == 0;
}

static_assert(ttl::tensor<float>);

static_assert(ttl::tensor<int>);
static_assert(ttl::tensor<int&>);
static_assert(ttl::tensor<int&&>);

static_assert(ttl::tensor<int const>);
static_assert(ttl::tensor<int const&>);
static_assert(ttl::tensor<int const&&>);

static_assert(ttl::tensor<int[3]>);
static_assert(ttl::tensor<int(&)[3]>);
static_assert(ttl::tensor<int(&&)[3]>);

static_assert(ttl::tensor<int const[3]>);
static_assert(ttl::tensor<int const(&)[3]>);
static_assert(ttl::tensor<int const(&&)[3]>);

static_assert(ttl::tensor<int[3][3]>);
static_assert(ttl::tensor<int(&)[3][3]>);
static_assert(ttl::tensor<int(&&)[3][3]>);

static_assert(ttl::tensor<int const[3][3]>);
static_assert(ttl::tensor<int const(&)[3][3]>);
static_assert(ttl::tensor<int const(&&)[3][3]>);

static_assert(ttl::tensor<std::span<int, 3>>);
static_assert(ttl::tensor<std::span<int[3], 3>>);
static_assert(ttl::tensor<std::span<int const, 3>>);
static_assert(ttl::tensor<std::span<int const[3], 3>>);

static_assert(ttl::tensor<std::array<int, 3>>);
static_assert(ttl::tensor<std::array<int[3], 3>>);
static_assert(ttl::tensor<std::array<int const, 3>>);
static_assert(ttl::tensor<std::array<int const[3], 3>>);

static_assert(ttl::tensor<std::mdspan<int, std::extents<std::size_t>>>);
static_assert(ttl::tensor<std::mdspan<int, std::extents<std::size_t, 3>>>);
static_assert(ttl::tensor<std::mdspan<int, std::extents<std::size_t, 3, std::dynamic_extent>>>);

static_assert(ttl::tensor<std::mdspan<int const, std::extents<std::size_t>>>);
static_assert(ttl::tensor<std::mdspan<int const, std::extents<std::size_t, 3>>>);
static_assert(ttl::tensor<std::mdspan<int const, std::extents<std::size_t, 3, std::dynamic_extent>>>);

static_assert(ttl::tensor<std::vector<int>>);
static_assert(ttl::tensor<std::vector<std::vector<int>>>);

static_assert(ttl::scalar<int>);
static_assert(ttl::scalar<int&>);
static_assert(ttl::scalar<int&&>);

static_assert(ttl::scalar<int const>);
static_assert(ttl::scalar<int const&>);
static_assert(ttl::scalar<int const&&>);

static_assert(ttl::scalar<float>);
static_assert(ttl::scalar<float&>);
static_assert(ttl::scalar<float&&>);

static_assert(ttl::scalar<float const>);
static_assert(ttl::scalar<float const&>);
static_assert(ttl::scalar<float const&&>);

static_assert(not ttl::expression<int[3]>);
static_assert(not ttl::expression<int(&)[3]>);
static_assert(not ttl::expression<int(&&)[3]>);

static_assert(not ttl::expression<int[3][3]>);
static_assert(not ttl::expression<int(&)[3][3]>);
static_assert(not ttl::expression<int(&&)[3][3]>);
