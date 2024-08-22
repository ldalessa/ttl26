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
        static_assert(_check_contracted_extents_static<_index, extents_type<A>>);

        using scalar_type = ttl::scalar_type<A>;

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
            assert(_check_contracted_extents_dynamic<_index>(ttl::extents(_a)));
        }

        template <class I, class... Is>
            requires (std::integral<I> or ... or std::integral<Is>)
        constexpr bind(A a, I i, Is... is)
                : bind(a, index(i), index(is)...)
        {
        }

        static constexpr auto rank = std::integral_constant<std::size_t, _outer.rank()>();

        static constexpr auto outer() {
            return _outer;
        }

        constexpr auto extents() const ->
            ARROW( select_extents<_index, _outer>(ttl::extents(_a)) );

        /// Innermost evaluation just remaps indices
        constexpr auto operator[](this auto&& self, std::integral auto... i) -> evaluate_type<A>
            requires (sizeof...(i) == _all.size())
        {
            return FWD(self)._evaluate(imap<_all, _index>, i...);
        }

        /// Need to inject the projected indices.
        constexpr auto operator[](this auto&& self, std::integral auto... i) -> evaluate_type<A>
            requires (_inner.size() <= sizeof...(i) and sizeof...(i) < _all.size())
        {
            return FWD(self)._project(self._projection_map(), i...);
        }

        /// Need to contract indices.
        constexpr auto operator[](this auto&& self, std::integral auto... i) -> scalar_type
            requires (rank <= sizeof...(i) and sizeof...(i) < _inner.size())
        {
            static constexpr std::size_t N = sizeof...(i);
            auto const contracted = select_extents<_inner, _index>(ttl::extents(self._a));
            std::size_t const e = contracted.extent(N);
            scalar_type accum {};
            for (std::size_t j = 0; j < e; ++j) {
                accum += self[i..., j];
            }
            return accum;
        }

      private:
        constexpr auto _projection_map() const {
            return _i.projection_map();
        }

        template <std::size_t... j>
        constexpr auto _evaluate(this auto&& self, std::index_sequence<j...>, std::integral auto... i) ->
            evaluate_type<A>
        {
            static_assert(sizeof...(i) == _all.size());
            static_assert(sizeof...(j) == _index.size());

            std::size_t const ind[] { std::size_t(i)... };
            return evaluate(FWD(self)._a, ind[j]...); /// @todo[c++26] i...[j]...
        }

        template <std::size_t... j>
        constexpr auto _project(this auto&& self, std::index_sequence<j...>, std::integral auto... i) ->
            evaluate_type<A>
        {
            static_assert(sizeof...(i) == _inner.size());
            return FWD(self)[i..., self._i[j]...];
        }
    };

    // Type deduction for binds.
    template <class A, istring... is>
    bind(A&, index<is>...) -> bind<A&, (istring{""} + ... + is)>;

    template <class A, istring... is>
    bind(A const&&, index<is>...) -> bind<A const, (istring{""} + ... + is)>;

    template <class A, class I, class... Is>
        requires (std::integral<I> or ... or std::integral<Is>)
    bind(A&, I, Is...) -> bind<A&, (to_istring<I> + ... + to_istring<Is>)>;

    template <class A, class I, class... Is>
        requires (std::integral<I> or ... or std::integral<Is>)
    bind(A&&, I, Is...) -> bind<A const, (to_istring<I> + ... + to_istring<Is>)>;
}

using namespace ttl::tree;

#undef DNDEBUG

static constexpr ttl::index<"i"> i;
static constexpr ttl::index<"j"> j;
static constexpr ttl::index<ttl::projection> p;

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
    static_assert(bind(b, 0).rank == 0);

    int c[3][3]{};
    static_assert(bind(c, i, j).rank == 2);
    static_assert(rank<decltype(bind(c, i, j))> == 2);
    static_assert(bind(c, 0, i).rank == 1);
    static_assert(bind(c, i, 0).rank == 1);
    static_assert(bind(c, 0, 0).rank == 0);

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

static constexpr bool check_bind_evaluate_plain()
{
    {
        int a{};
        bind A{a};

        assert(0 == A[]);
        assert(0 == evaluate(A));
        A[] = 1;
        assert(1 == a);
    }

    {
        int a[]{0, 1};
        bind A(a, i);
        assert(0 == A[0]);
        assert(1 == A[1]);
        assert(0 == evaluate(A, 0));
        assert(1 == evaluate(A, 1));
        A[0] = 1;
        A[1] = 2;
        assert(1 == a[0]);
        assert(2 == a[1]);

        bind B(std::as_const(a), i);
        assert(1 == B[0]);
        assert(2 == B[1]);
    }

    {
        int a[2][2]{{1, 2}, {3, 4}};
        bind A(a, i, j);
        assert((a[0][0] == A[0,0]));
        assert((a[0][1] == A[0,1]));
        assert((a[1][0] == A[1,0]));
        assert((a[1][1] == A[1,1]));

        A[0,0] = 2;
        A[0,1] = 3;
        A[1,0] = 4;
        A[1,1] = 5;

        assert(2 == a[0][0]);
        assert(3 == a[0][1]);
        assert(4 == a[1][0]);
        assert(5 == a[1][1]);

        bind B(std::as_const(a), i, j);
        assert((2 == B[0,0]));
        assert((3 == B[0,1]));
        assert((4 == B[1,0]));
        assert((5 == B[1,1]));
    }

    return true;
}

static constexpr bool check_bind_evaluate_projection()
{
    {
        int a[]{0, 1};
        bind A(a, 0);
        assert(0 == A[]);
        assert(0 == evaluate(A));
        A[] = 1;
        assert(1 == a[0]);

        bind B(std::as_const(a), 1);
        assert(1 == B[0]);
    }

    {
        int a[2][2]{{1, 2}, {3, 4}};

        bind A(a, i, 0);
        assert((a[0][0] == A[0]));
        assert((a[1][0] == A[1]));

        bind B(a, 0, i);
        assert((a[0][0] == B[0]));
        assert((a[0][1] == B[1]));

        bind C(a, i, 1);
        assert((a[0][1] == C[0]));
        assert((a[1][1] == C[1]));

        bind D(a, 1, i);
        assert((a[1][0] == D[0]));
        assert((a[1][1] == D[1]));
    }

    return true;
}

static constexpr bool check_bind_evaluate_contraction()
{
    int a[2][2]{{1, 2}, {3, 4}};
    bind A(a, i ,i);
    assert(A.rank() == 0);
    assert(5 == A[]);

    int b[1][2][2]{{{1, 2}, {3, 4}}};
    bind B(b, i, j, j);
    assert(B.rank() == 1);
    assert(5 == B[0]);

    std::vector c{std::vector<int>{1,2}, std::vector<int>{3,4}};
    bind C(c, i, i);
    assert(C.rank() == 0);
    assert(5 == C[]);

    return true;
}

static constexpr bool check_bind_evaluate_scalar()
{
    int a = 0;
    bind A(a);
    assert(0 == A);

    int b[2][2]{};
    bind B(b, i, i);
    assert(0 == B);

    return true;
}

static_assert(check_bind_ctad());
static_assert(check_bind_extents());
static_assert(check_bind_evaluate_plain());
static_assert(check_bind_evaluate_projection());
static_assert(check_bind_evaluate_contraction());
static_assert(check_bind_evaluate_scalar());
