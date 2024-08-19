module;

#include <cstdio>
#include <utility>

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
        static constexpr auto _rank = _outer.rank();

        A _a;
        index<_index> _i{};

        constexpr bind(A a, ttl::index<_index> i)
                : _a(a)
                , _i(i)
        {}

        constexpr bind(A a) requires (_index.n_projected() == 0)
                : _a(a)
        {}

        static constexpr auto outer() {
            return _outer;
        }

        static constexpr auto rank() -> std::size_t {
            return _rank;
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
    using ttl::index;

    int a = 0;
    bind _{a};
    bind _{std::as_const(a)};
    bind _{std::move(a)};
    bind _{std::move(std::as_const(a))};

    // int x[3]{};
    // bind<int(&)[3], "i"> b(x);
    return true;
}

static_assert(check_bind());
