#undef DNDEBUG
#include <ttl/tensor/extents.hpp>

using namespace ttl;
using namespace ttl::concepts;

static_assert(has_extents<int>);
static_assert(has_extents<int&>);
static_assert(has_extents<int&&>);

static_assert(has_extents<int const>);
static_assert(has_extents<int const&>);
static_assert(has_extents<int const&&>);

static_assert(has_extents<int[3]>);
static_assert(has_extents<int(&)[3]>);
static_assert(has_extents<int(&&)[3]>);

static_assert(has_extents<int const[3]>);
static_assert(has_extents<int const(&)[3]>);
static_assert(has_extents<int const(&&)[3]>);

static_assert(has_extents<int[3][3]>);
static_assert(has_extents<int(&)[3][3]>);
static_assert(has_extents<int(&&)[3][3]>);

static_assert(has_extents<int const[3]>);
static_assert(has_extents<int const(&)[3]>);
static_assert(has_extents<int const(&&)[3]>);

static_assert(has_extents<float>);

static_assert(has_extents<std::span<int, 3>>);
static_assert(has_extents<std::span<int const, 3>>);
static_assert(has_extents<std::span<int[3], 3>>);
static_assert(has_extents<std::span<int[3][3], 3>>);

static_assert(has_extents<std::array<int, 3>>);
static_assert(has_extents<std::array<int, 3>&>);
static_assert(has_extents<std::array<int, 3>&&>);

static_assert(has_extents<std::array<int, 3> const>);
static_assert(has_extents<std::array<int, 3> const&>);
static_assert(has_extents<std::array<int, 3> const&&>);

static_assert(has_extents<std::array<int const, 3>>);
static_assert(has_extents<std::array<int const, 3>&>);
static_assert(has_extents<std::array<int const, 3>&&>);

static_assert(has_extents<std::array<int const, 3> const>);
static_assert(has_extents<std::array<int const, 3> const&>);
static_assert(has_extents<std::array<int const, 3> const&&>);

static_assert(has_extents<std::array<int[3], 3>>);
static_assert(has_extents<std::array<int[3][3], 3>&>);
static_assert(has_extents<std::array<int[3][3][3], 3>&>);

static_assert(has_extents<std::array<int const[3], 3>>);
static_assert(has_extents<std::array<int const[3][3], 3>&>);
static_assert(has_extents<std::array<int const[3][3][3], 3>&>);

static_assert(has_extents<std::vector<int>>);
static_assert(has_extents<std::vector<int>&>);
static_assert(has_extents<std::vector<int>&&>);

static_assert(has_extents<std::vector<int> const>);
static_assert(has_extents<std::vector<int> const&>);
static_assert(has_extents<std::vector<int> const&&>);

static_assert(has_extents<std::vector<std::vector<int>>>);
static_assert(has_extents<std::vector<std::vector<int>>&>);
static_assert(has_extents<std::vector<std::vector<int>>&&>);

static_assert(has_extents<std::vector<std::vector<int>> const>);
static_assert(has_extents<std::vector<std::vector<int>> const&>);
static_assert(has_extents<std::vector<std::vector<int>> const&&>);

static_assert(has_extents<std::vector<std::array<int, 3>>>);
static_assert(has_extents<std::vector<std::array<int[3], 3>>>);

static_assert(has_extents<std::mdspan<int, std::extents<std::size_t>>>);
static_assert(has_extents<std::mdspan<int, std::extents<std::size_t, 1>>>);
static_assert(has_extents<std::mdspan<int, std::extents<std::size_t, 2, 3>>>);

static_assert(not has_extents<std::extents<std::size_t>>);
static_assert(not is_static_extent<std::extents<std::size_t, 1>, 0>);

static constexpr bool check_extents()
{
	static constexpr std::extents<std::size_t> scalar;
	static constexpr std::extents<std::size_t, 3> vector_3;
	static constexpr std::extents<std::size_t, 3, 3> vector_3_3;

	static constexpr std::extents<std::size_t, std::dynamic_extent> vector_n3(3);
	static constexpr std::extents<std::size_t, std::dynamic_extent, 3> vector_n3_3(3, 3);
	static constexpr std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent> vector_n3_n3(3, 3);

	assert(ttl::extents(static_cast<int>(1)) == scalar);
	assert(ttl::extents(static_cast<int const>(1)) == scalar);
	assert(ttl::extents(static_cast<int const&&>(1)) == scalar);

	int a = 1;
	assert(ttl::extents(static_cast<int&>(a)) == scalar);
	assert(ttl::extents(static_cast<int&&>(a)) == scalar);
	assert(ttl::extents(static_cast<int const&>(a)) == scalar);
	assert(ttl::extents(static_cast<int const&&>(a)) == scalar);

	int b[3]{};
	assert(ttl::extents(std::span(b)) == vector_3);
	assert(ttl::extents(std::span(std::as_const(b))) == vector_3);

	assert(ttl::extents(std::mdspan(b, 3)) == vector_n3);
	assert(ttl::extents(std::mdspan(std::as_const(b), 3)) == vector_n3);

	assert(ttl::extents(static_cast<int(&)[3]>(b)) == vector_3);
	assert(ttl::extents(static_cast<int(&&)[3]>(b)) == vector_3);
	assert(ttl::extents(static_cast<int const(&)[3]>(b)) == vector_3);
	assert(ttl::extents(static_cast<int const(&&)[3]>(b)) == vector_3);

	int c[3][3]{};
	assert(ttl::extents(std::span(c)) == vector_3_3);
	assert(ttl::extents(std::span(std::as_const(c))) == vector_3_3);

	assert(ttl::extents(std::mdspan(b, 3, 3)) == vector_n3_n3);
	assert(ttl::extents(std::mdspan(std::as_const(b), 3, 3)) == vector_n3_n3);

	assert(ttl::extents(static_cast<int(&)[3][3]>(c)) == vector_3_3);
	assert(ttl::extents(static_cast<int(&&)[3][3]>(c)) == vector_3_3);
	assert(ttl::extents(static_cast<int const(&)[3][3]>(c)) == vector_3_3);
	assert(ttl::extents(static_cast<int const(&&)[3][3]>(c)) == vector_3_3);

	std::array<int, 3> d{};
	assert(ttl::extents(std::span(d)) == vector_3);
	assert(ttl::extents(std::span(std::as_const(d))) == vector_3);

	assert(ttl::extents(d) == vector_3);
	assert(ttl::extents(std::as_const(d)) == vector_3);
	assert(ttl::extents(std::move(d)) == vector_3);

	std::array<int const, 3> e{};
	assert(ttl::extents(std::span(e)) == vector_3);
	assert(ttl::extents(std::span(std::as_const(e))) == vector_3);

	assert(ttl::extents(e) == vector_3);
	assert(ttl::extents(std::as_const(e)) == vector_3);
	assert(ttl::extents(std::move(e)) == vector_3);

	std::array<std::array<int, 3>, 3> f{};
	assert(ttl::extents(std::span(f)) == vector_3_3);
	assert(ttl::extents(std::span(std::as_const(f))) == vector_3_3);

	assert(ttl::extents(f) == vector_3_3);
	assert(ttl::extents(std::as_const(f)) == vector_3_3);
	assert(ttl::extents(std::move(f)) == vector_3_3);

	std::vector<int> g(3);
	assert(ttl::extents(std::span(g)) == vector_n3);
	assert(ttl::extents(std::span(std::as_const(g))) == vector_n3);

	assert(ttl::extents(g) == vector_n3);
	assert(ttl::extents(std::as_const(g)) == vector_n3);
	assert(ttl::extents(std::move(g)) == vector_n3);

	std::vector<std::array<int, 3>> h(3);
	assert(ttl::extents(std::span(h)) == vector_n3_3);
	assert(ttl::extents(std::span(std::as_const(h))) == vector_n3_3);

	assert(ttl::extents(h) == vector_n3_3);
	assert(ttl::extents(std::as_const(h)) == vector_n3_3);
	assert(ttl::extents(std::move(h)) == vector_n3_3);

	return true;
}

static_assert(check_extents());
