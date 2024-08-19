module;

#include <array>
#include <cassert>
#include <cstdio>
#include <mdspan>
#include <utility>
#include <vector>

module ttl:bind;
import :concepts;
import :expression;
import :index;
import :tensor;

namespace ttl::tree
{
    template <tensor A, istring _index>
    struct bind : expression
    {
        static_assert(rank<A> == _index.size());

        static constexpr auto _outer = _index.outer();
        static constexpr auto _inner = _index.inner();
        static constexpr auto _all = _index.all();

        A _a;
        index<_index> _i{};

        template <istring... _>
        constexpr bind(A a, index<_>... is)
                : _a(a)
                , _i((index<"">{} + ... + is))
        {
            static_assert((istring{""} + ... + _) == _index);
        }

        static constexpr auto rank = rank_v<_outer.rank()>;

        static constexpr auto outer() {
            return _outer;
        }

        constexpr auto extents() const ->
            ARROW( select_extents<_index, _outer>(ttl::extents(_a)) );
    };

    // Type deduction for binds.
    template <class A, istring... is>
    bind(A&, index<is>...) -> bind<A&, (istring{""} + ... + is)>;

    template <class A, istring... is>
    bind(A const&&, index<is>...) -> bind<A const, (istring{""} + ... + is)>;
}

using namespace ttl::tree;

#undef DNDEBUG

static constexpr ttl::index<"i"> i;
static constexpr ttl::index<"j"> j;

static constexpr bool check_bind_ctad()
{
    int a{};
    bind _{a};
    bind _{std::as_const(a)};
    bind _{std::move(a)};
    bind _{std::move(std::as_const(a))};

    int b[3]{};
    bind _{b, i};
    bind _{std::as_const(b), i};

    std::array c = { 1, 2, 3 };
    bind _{c, i};
    bind _{std::as_const(c), i};
    bind _{std::move(c), i };
    bind _{std::move(std::as_const(c)), i };

    std::vector d = { 1, 2, 3 };
    bind _{d, i};
    bind _{std::as_const(d), i};
    bind _{std::move(d), i };
    bind _{std::move(std::as_const(d)), i };

    int e[3][3]{};
    bind _{e, i ,j};
    bind _{std::as_const(e), i, j};

    std::array<std::array<int, 3>, 3> f{};
    bind _{f, i, j};
    bind _{std::as_const(f), i, j};
    bind _{std::move(f), i, j};
    bind _{std::move(std::as_const(f)), i, j};

    bind _{f, i + j };
    bind _{std::as_const(f), i + j };
    bind _{std::move(f), i + j };
    bind _{std::move(std::as_const(f)), i + j };

    return true;
}

static constexpr bool check_bind_rank()
{
    int a{};
    static_assert(bind(a).rank == 0);
    static_assert(rank<decltype(bind(a))> == 0);

    int b[3]{};
    static_assert(bind(b, i).rank == 1);
    static_assert(rank<decltype(bind(b, i))> == 1);

    int c[3][3]{};
    static_assert(bind(c, i, j).rank == 2);
    static_assert(rank<decltype(bind(c, i, j))> == 2);

    return true;
}

static constexpr bool check_bind_extents()
{
    int a{};
    bind ba{a};
    assert(ba.extents() == std::extents<std::size_t>{});
    assert(ttl::extents(ba) == std::extents<std::size_t>{});

    int b[3]{};
    bind bb(b, i);
    assert((bb.extents() == std::extents<std::size_t, 3>{}));
    assert((ttl::extents(bb) == std::extents<std::size_t, 3>{}));
    static_assert(ttl::extent<0>(bb) == 3);

    int c[3][7]{};
    bind bc(c, i, j);
    assert((bc.extents() == std::extents<std::size_t, 3, 7>{}));
    assert((ttl::extents(bc) == std::extents<std::size_t, 3, 7>{}));
    static_assert(ttl::extent<0>(bc) == 3);
    static_assert(ttl::extent<1>(bc) == 7);

    std::vector d = { 1, 2, 3 };
    bind bd(d, i);
    assert((bd.extents() == std::extents<std::size_t, std::dynamic_extent>{3}));
    assert((ttl::extents(bd) == std::extents<std::size_t, std::dynamic_extent>{3}));
    assert(ttl::extent<0>(bd) == 3);

    std::vector e = {
        std::array{1,2,3},
        std::array{4,5,6}
    };
    bind be(e, i, j);
    assert((be.extents() == std::extents<std::size_t, std::dynamic_extent, 3>{2}));
    assert(ttl::extent<0>(be) == 2);
    static_assert(ttl::extent<1>(be) == 3);

    return true;
}

static_assert(check_bind_ctad());
static_assert(check_bind_extents());
