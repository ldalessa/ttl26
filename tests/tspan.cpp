#undef DNDEBUG

#include <ttl/tspan.hpp>

static_assert(ttl::concepts::scalar<ttl::tspan<int, std::extents<std::size_t>>>);
static_assert(ttl::concepts::tensor<ttl::tspan<int, std::extents<std::size_t, 3>>>);
static_assert(ttl::concepts::tensor<ttl::tspan<int, std::extents<std::size_t, 3, 3>>>);
static_assert(ttl::concepts::tensor<ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>>>);

static constexpr bool check_tspan_c_array()
{
	int a[16]{};

	ttl::tspan<int, std::extents<std::size_t, 16>> _(a);
	ttl::tspan<int, std::extents<std::size_t, 4, 4>> _(a);
	ttl::tspan<int, std::extents<std::size_t, 2, 2, 2, 2>> _(a);

	ttl::tspan<int, std::extents<std::size_t, 16>> _(a, 16);
	ttl::tspan<int, std::extents<std::size_t, 4, 4>> _(a, 4, 4);
	ttl::tspan<int, std::extents<std::size_t, 2, 2, 2, 2>> _(a, 2, 2, 2, 2);

	ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent>> _(a, 16);
	ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>> _(a, 4, 4);
	ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>> _(a, 2, 2, 2, 2);

	ttl::tspan _(a);

	ttl::tspan _(a, std::extents<std::size_t, 16>());
	ttl::tspan _(a, std::extents<std::size_t, 4, 4>());
	ttl::tspan _(a, std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(a, 16);
	ttl::tspan _(a, 4, 4);
	ttl::tspan _(a, 2, 2, 2, 2);

	auto b = std::span(a);

	ttl::tspan<int, std::extents<std::size_t, 16>> _(b);
	// ttl::tspan<int, std::extents<std::size_t, 4, 4>> _(b); @fixme constructor
	// ttl::tspan<int, std::extents<std::size_t, 2, 2, 2, 2>> _(b); @fixme constructor

	ttl::tspan<int, std::extents<std::size_t, 16>> _(b, 16);
	ttl::tspan<int, std::extents<std::size_t, 4, 4>> _(b, 4, 4);
	ttl::tspan<int, std::extents<std::size_t, 2, 2, 2, 2>> _(b, 2, 2, 2, 2);

	ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent>> _(b, 16);
	ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>> _(b, 4, 4);
	ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>> _(b, 2, 2, 2, 2);

	ttl::tspan _(b);

	ttl::tspan _(b, std::extents<std::size_t, 16>());
	ttl::tspan _(b, std::extents<std::size_t, 4, 4>());
	ttl::tspan _(b, std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(b, 16);
	ttl::tspan _(b, 4, 4);
	ttl::tspan _(b, 2, 2, 2, 2);

	auto c = std::span(a, 16);
	ttl::tspan _(c);

	ttl::tspan _(c, std::extents<std::size_t, 16>());
	ttl::tspan _(c, std::extents<std::size_t, 4, 4>());
	ttl::tspan _(c, std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(c, 16);
	ttl::tspan _(c, 4, 4);
	ttl::tspan _(c, 2, 2, 2, 2);

	int const d[16]{};
	ttl::tspan _(d);

	ttl::tspan _(d, std::extents<std::size_t, 16>());
	ttl::tspan _(d, std::extents<std::size_t, 4, 4>());
	ttl::tspan _(d, std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(d, 16);
	ttl::tspan _(d, 4, 4);
	ttl::tspan _(d, 2, 2, 2, 2);

	auto e = std::span(b);
	ttl::tspan _(e);

	ttl::tspan _(e, std::extents<std::size_t, 16>());
	ttl::tspan _(e, std::extents<std::size_t, 4, 4>());
	ttl::tspan _(e, std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(e, 16);
	ttl::tspan _(e, 4, 4);
	ttl::tspan _(e, 2, 2, 2, 2);

	auto f = std::span(b);
	ttl::tspan _(f);

	ttl::tspan _(f, std::extents<std::size_t, 16>());
	ttl::tspan _(f, std::extents<std::size_t, 4, 4>());
	ttl::tspan _(f, std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(f, 16);
	ttl::tspan _(f, 4, 4);
	ttl::tspan _(f, 2, 2, 2, 2);

	return true;
}

static constexpr bool check_tspan_array()
{
	std::array<int, 16> a{};

	ttl::tspan<int, std::extents<std::size_t, 16>> _(a);
	// ttl::tspan<int, std::extents<std::size_t, 4, 4>> _(a); @fixme constructor
	// ttl::tspan<int, std::extents<std::size_t, 2, 2, 2, 2>> _(a); @fixme constructor

	ttl::tspan<int, std::extents<std::size_t, 16>> _(a, 16);
	ttl::tspan<int, std::extents<std::size_t, 4, 4>> _(a, 4, 4);
	ttl::tspan<int, std::extents<std::size_t, 2, 2, 2, 2>> _(a, 2, 2, 2, 2);

	ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent>> _(a, 16);
	ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>> _(a, 4, 4);
	ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>> _(a, 2, 2, 2, 2);

	ttl::tspan _(a);

	ttl::tspan _(a, std::extents<std::size_t, 16>());
	ttl::tspan _(a, std::extents<std::size_t, 4, 4>());
	ttl::tspan _(a, std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(a, 16);
	ttl::tspan _(a, 4, 4);
	ttl::tspan _(a, 2, 2, 2, 2);

	ttl::tspan _(a.begin(), std::extents<std::size_t, 16>());
	ttl::tspan _(a.begin(), std::extents<std::size_t, 4, 4>());
	ttl::tspan _(a.begin(), std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(a.begin(), 16);
	ttl::tspan _(a.begin(), 4, 4);
	ttl::tspan _(a.begin(), 2, 2, 2, 2);

	ttl::tspan _(a.cbegin(), std::extents<std::size_t, 16>());
	ttl::tspan _(a.cbegin(), std::extents<std::size_t, 4, 4>());
	ttl::tspan _(a.cbegin(), std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(a.cbegin(), 16);
	ttl::tspan _(a.cbegin(), 4, 4);
	ttl::tspan _(a.cbegin(), 2, 2, 2, 2);

	const std::array<int, 16> b{};
	ttl::tspan _(b);

	ttl::tspan _(b, std::extents<std::size_t, 16>());
	ttl::tspan _(b, std::extents<std::size_t, 4, 4>());
	ttl::tspan _(b, std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(b, 16);
	ttl::tspan _(b, 4, 4);
	ttl::tspan _(b, 2, 2, 2, 2);

	ttl::tspan _(b.begin(), std::extents<std::size_t, 16>());
	ttl::tspan _(b.begin(), std::extents<std::size_t, 4, 4>());
	ttl::tspan _(b.begin(), std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(b.begin(), 16);
	ttl::tspan _(b.begin(), 4, 4);
	ttl::tspan _(b.begin(), 2, 2, 2, 2);

	auto c = std::span(b);
	ttl::tspan _(c);

	ttl::tspan _(c, std::extents<std::size_t, 16>());
	ttl::tspan _(c, std::extents<std::size_t, 4, 4>());
	ttl::tspan _(c, std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(c, 16);
	ttl::tspan _(c, 4, 4);
	ttl::tspan _(c, 2, 2, 2, 2);

	ttl::tspan _(c.begin(), std::extents<std::size_t, 16>());
	ttl::tspan _(c.begin(), std::extents<std::size_t, 4, 4>());
	ttl::tspan _(c.begin(), std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(c.begin(), 16);
	ttl::tspan _(c.begin(), 4, 4);
	ttl::tspan _(c.begin(), 2, 2, 2, 2);

	std::array<int const, 16> d{};
	ttl::tspan _(d);

	ttl::tspan _(d, std::extents<std::size_t, 16>());
	ttl::tspan _(d, std::extents<std::size_t, 4, 4>());
	ttl::tspan _(d, std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(d, 16);
	ttl::tspan _(d, 4, 4);
	ttl::tspan _(d, 2, 2, 2, 2);

	ttl::tspan _(d.begin(), std::extents<std::size_t, 16>());
	ttl::tspan _(d.begin(), std::extents<std::size_t, 4, 4>());
	ttl::tspan _(d.begin(), std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(d.begin(), 16);
	ttl::tspan _(d.begin(), 4, 4);
	ttl::tspan _(d.begin(), 2, 2, 2, 2);

	ttl::tspan _(d.cbegin(), std::extents<std::size_t, 16>());
	ttl::tspan _(d.cbegin(), std::extents<std::size_t, 4, 4>());
	ttl::tspan _(d.cbegin(), std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(d.cbegin(), 16);
	ttl::tspan _(d.cbegin(), 4, 4);
	ttl::tspan _(d.cbegin(), 2, 2, 2, 2);

	const std::array<int const, 16> e{};
	ttl::tspan _(e);

	ttl::tspan _(e, std::extents<std::size_t, 16>());
	ttl::tspan _(e, std::extents<std::size_t, 4, 4>());
	ttl::tspan _(e, std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(e, 16);
	ttl::tspan _(e, 4, 4);
	ttl::tspan _(e, 2, 2, 2, 2);

	ttl::tspan _(e.begin(), std::extents<std::size_t, 16>());
	ttl::tspan _(e.begin(), std::extents<std::size_t, 4, 4>());
	ttl::tspan _(e.begin(), std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(e.begin(), 16);
	ttl::tspan _(e.begin(), 4, 4);
	ttl::tspan _(e.begin(), 2, 2, 2, 2);

	return true;
}

static constexpr bool check_tspan_vector()
{
	std::vector<int> a(16);
	ttl::tspan _(a);

	ttl::tspan _(a, std::extents<std::size_t, 16>());
	ttl::tspan _(a, std::extents<std::size_t, 4, 4>());
	ttl::tspan _(a, std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(a, 16);
	ttl::tspan _(a, 4, 4);
	ttl::tspan _(a, 2, 2, 2, 2);

	ttl::tspan _(a.begin(), std::extents<std::size_t, 16>());
	ttl::tspan _(a.begin(), std::extents<std::size_t, 4, 4>());
	ttl::tspan _(a.begin(), std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(a.begin(), 16);
	ttl::tspan _(a.begin(), 4, 4);
	ttl::tspan _(a.begin(), 2, 2, 2, 2);

	ttl::tspan _(a.cbegin(), std::extents<std::size_t, 16>());
	ttl::tspan _(a.cbegin(), std::extents<std::size_t, 4, 4>());
	ttl::tspan _(a.cbegin(), std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(a.cbegin(), 16);
	ttl::tspan _(a.cbegin(), 4, 4);
	ttl::tspan _(a.cbegin(), 2, 2, 2, 2);

	const std::vector<int> b(16);
	ttl::tspan _(b);

	ttl::tspan _(b, std::extents<std::size_t, 16>());
	ttl::tspan _(b, std::extents<std::size_t, 4, 4>());
	ttl::tspan _(b, std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(b, 16);
	ttl::tspan _(b, 4, 4);
	ttl::tspan _(b, 2, 2, 2, 2);

	ttl::tspan _(b.begin(), std::extents<std::size_t, 16>());
	ttl::tspan _(b.begin(), std::extents<std::size_t, 4, 4>());
	ttl::tspan _(b.begin(), std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(b.begin(), 16);
	ttl::tspan _(b.begin(), 4, 4);
	ttl::tspan _(b.begin(), 2, 2, 2, 2);

	return true;
}

static constexpr bool check_tspan_new()
{
	int *a = new int[16]{};

	ttl::tspan<int, std::extents<std::size_t, 16>> _(a);
	ttl::tspan<int, std::extents<std::size_t, 4, 4>> _(a);
	ttl::tspan<int, std::extents<std::size_t, 2, 2, 2, 2>> _(a);

	ttl::tspan<int, std::extents<std::size_t, 16>> _(a, 16);
	ttl::tspan<int, std::extents<std::size_t, 4, 4>> _(a, 4, 4);
	ttl::tspan<int, std::extents<std::size_t, 2, 2, 2, 2>> _(a, 2, 2, 2, 2);

	ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent>> _(a, 16);
	ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>> _(a, 4, 4);
	ttl::tspan<int, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>> _(a, 2, 2, 2, 2);

	// ttl::tspan _(b); // expected failure

	ttl::tspan _(a, std::extents<std::size_t, 16>());
	ttl::tspan _(a, std::extents<std::size_t, 4, 4>());
	ttl::tspan _(a, std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(a, 16);
	ttl::tspan _(a, 4, 4);
	ttl::tspan _(a, 2, 2, 2, 2);

	delete [] a;

	int const *b = new int const[16]{};
	// ttl::tspan _(b); // expected failure

	ttl::tspan _(b, std::extents<std::size_t, 16>());
	ttl::tspan _(b, std::extents<std::size_t, 4, 4>());
	ttl::tspan _(b, std::extents<std::size_t, 2, 2, 2, 2>());

	ttl::tspan _(b, 16);
	ttl::tspan _(b, 4, 4);
	ttl::tspan _(b, 2, 2, 2, 2);

	delete [] b;

	return true;
}

static constexpr bool check_tspan_bind()
{
	ttl::index<"i"> i;
	ttl::index<"j"> j;

	int a[16]{};
	ttl::tspan A(a, 4, 4);
	ttl::tree::bind B = A(i,j);
	return true;
}

static_assert(check_tspan_c_array());
static_assert(check_tspan_array());
static_assert(check_tspan_vector());
static_assert(check_tspan_new());
static_assert(check_tspan_bind());

