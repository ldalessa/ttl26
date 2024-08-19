module;

#include <array>
#include <cstdio>
#include <utility>
#include <vector>

module ttl:bind;
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
    };

    // Type deduction for binds.
    template <class A, istring... is>
    bind(A&, index<is>...) -> bind<A&, (istring<1>{} + ... + is)>;

    template <class A, istring... is>
    bind(A const&&, index<is>...) -> bind<A const, (istring<1>{} + ... + is)>;
}

using namespace ttl::tree;

#undef DNDEBUG

static constexpr bool check_bind()
{
    ttl::index<"i"> i;
    ttl::index<"j"> j;

    int a = 0;
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

static_assert(check_bind());
