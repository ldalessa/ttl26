#undef DNDEBUG

#include <ttl/index/cstring.hpp>

using namespace ttl;
using namespace ttl::concepts;

template <class T, class U>
static constexpr auto cstring_indexable_with = requires (T t, U u) {
	t.count(u);
	t.index_of_1(u);
};

static constexpr bool test_cstring()
{
	constexpr cstring i = "i";
	static_assert(cstring_indexable_with<decltype(i), char>);
	static_assert(cstring_indexable_with<decltype(i), char8_t>);
	static_assert(cstring_indexable_with<decltype(i), char16_t>);
	static_assert(not cstring_indexable_with<decltype(i), char32_t>);
	static_assert(not cstring_indexable_with<decltype(i), wchar_t>);
	assert(i.size() == 1);
	assert(i[0] == 'i');
	assert(i[1] == '\0');
	assert(i.index_of_1('i') == 0);
	assert(i.index_of_1('j') == i.size());

	constexpr auto ii = i + i;
	assert(ii.size() == 2);
	assert(ii[0] == 'i');
	assert(ii[1] == 'i');
	assert(ii.index_of_1('i') == 0);
	assert(ii.index_of_1('j') == ii.size());
	assert(ii.index_of_2('i')[0] == 0);
	assert(ii.index_of_2('i')[1] == 1);

	constexpr auto jiik = cstring("j") + ii + cstring("k");
	assert(jiik.size() == 4);
	assert(jiik[0] == 'j');
	assert(jiik[1] == 'i');
	assert(jiik[2] == 'i');
	assert(jiik[3] == 'k');
	assert(jiik.index_of_1('j') == 0);
	assert(jiik.index_of_1('i') == 1);
	assert(jiik.index_of_1('k') == 3);
	assert(jiik.index_of_1('l') == jiik.size());
	assert(jiik.index_of_2('i')[0] == 1);
	assert(jiik.index_of_2('i')[1] == 2);

	constexpr cstring μ = u"μ";
	static_assert(cstring_indexable_with<decltype(μ), char>);
	static_assert(cstring_indexable_with<decltype(μ), char8_t>);
	static_assert(cstring_indexable_with<decltype(μ), char16_t>);
	static_assert(not cstring_indexable_with<decltype(μ), char32_t>);
	static_assert(not cstring_indexable_with<decltype(μ), wchar_t>);
	assert(μ.size() == 1);
	assert(μ[0] == u'μ');
	assert(μ.index_of_1(u'μ') == 0);

	constexpr cstring test = "test";
	constexpr std::array map = index_of<test, 't'>;
	assert(map.size() == 2);
	assert(map[0] == 0);
	assert(map[1] == 3);

	return true;
}

static_assert(test_cstring());
