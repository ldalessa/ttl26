module;

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>

module ttl:cstring;

namespace stdr = std::ranges;

namespace ttl
{
    namespace concepts
    {
        template <class T>
        concept character = std::same_as<T, char16_t>
            or std::same_as<T, char8_t>
            or std::same_as<T, char>;
    }

    /// A basic template for a constexpr c-string.
    template <std::size_t N>
    struct cstring
    {
        using char_t = char16_t;

        char_t _data[N]{};

        consteval cstring() = default;
        consteval cstring(concepts::character auto const (&str)[N]) {
            _strcpy(str, _data);
        }

        template <std::size_t M>
        constexpr bool operator==(cstring<M> const& b) const {
            return stdr::equal(*this, b);
        }

        /// Explicit metafunction specialization for the concat type.
        /// @{
        template <class>
        struct _concat_type;

        template <template <std::size_t> class string>
        struct _concat_type<string<N>> {
            template <std::size_t M>
            using type = string<N + M -1>;
        };

        template <class U, std::size_t M>
        using _concat_type_t = _concat_type<U>::template type<M>;
        /// @}

        /// String concatenation.
        ///
        /// This uses the explicit object parameter and the concat_type
        /// metafunction to support all subclass types.
        template <class U, std::size_t M>
        constexpr auto operator+(this U const& self, cstring<M> const& b)
            -> _concat_type_t<U, M>
        {
            _concat_type_t<U, M> out;
            _strcpy(b, _strcpy(self, out.begin()));
            return out;
        }

        constexpr auto size() const -> std::size_t {
            return index_of_1(u'\0');
        }

        constexpr auto begin(this auto&& self) -> auto* {
            return FWD(self)._data;
        }

        struct _end_sentinel {
            constexpr bool operator==(char_t const* const str) const {
                return *str == 0;
            }
        };

        static constexpr auto end() -> _end_sentinel {
            return {};
        }

        constexpr auto operator[](this auto&& self, std::size_t const i) -> auto& {
            return FWD(self)._data[i];
        }

        constexpr auto count(concepts::character auto const c) const ->
            std::size_t
        {
            return stdr::count(*this, c);
        }

        constexpr auto index_of_1(concepts::character auto const c) const ->
            std::size_t
        {
            return stdr::find(*this, c) - _data;
        }

        constexpr auto index_of_1_nth(concepts::character auto const c, std::size_t const n) const ->
            std::size_t
        {
            std::size_t i = 0;
            return stdr::find_if(*this, [&](char_t const d) {
                return (c == d) and (i++ == n);
            }) - _data;
        }

        struct _index_of_2_result
        {
            std::ptrdiff_t _data[2]{};
            constexpr auto operator[](std::size_t const i) const -> std::size_t {
                return _data[i];
            }
        };

        constexpr auto index_of_2(concepts::character auto const c) const ->
            _index_of_2_result
        {
            auto const i = stdr::find(*this, c);
            auto const j = stdr::find(i + 1, end(), c);
            return {
                i - _data,
                j - _data
            };
        }

      private:
        /// No constexpr std::strcpy or __builtin_strcpy, so implement one
        /// ranges-style.
        static constexpr auto _strcpy(stdr::contiguous_range auto&& in, char_t *out) -> char_t*
        {
            auto i = stdr::begin(in);
            while (*i != 0) {
                *out++ = *i++;
            }
            return out;
        }
    };
}

using namespace ttl;
using namespace ttl::concepts;

#undef DNDEBUG

template <class T, class U>
static constexpr auto cstring_indexable_with = requires (T t, U u) {
    t.count(u);
    t.index_of_1(u);
    t.index_of_2(u);
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
    assert(i.index_of_2('i')[0] == 0);
    assert(i.index_of_2('i')[1] == i.size());

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
    assert(jiik.index_of_2('j')[0] == 0);
    assert(jiik.index_of_2('i')[0] == 1);
    assert(jiik.index_of_2('i')[1] == 2);
    assert(jiik.index_of_2('k')[0] == 3);

    constexpr cstring μ = u"μ";
    static_assert(cstring_indexable_with<decltype(μ), char>);
    static_assert(cstring_indexable_with<decltype(μ), char8_t>);
    static_assert(cstring_indexable_with<decltype(μ), char16_t>);
    static_assert(not cstring_indexable_with<decltype(μ), char32_t>);
    static_assert(not cstring_indexable_with<decltype(μ), wchar_t>);
    assert(μ.size() == 1);
    assert(μ[0] == u'μ');
    assert(μ.index_of_1(u'μ') == 0);

    return true;
}

static_assert(test_cstring());
