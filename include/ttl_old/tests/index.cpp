#undef NDEBUG

#include <ttl/index_string.hpp>

#include <cassert>

static constexpr bool _tests()
{
    int p = 0; // used in some index_of tests

    ttl::index_string _ {};
    ttl::index_string i { "i" };

    assert(i == i + _);
    assert(i == i);
    assert(i.size() == 1);
    assert(i.rank() == 1);
    assert(i.outer() == i);
    assert(i.inner() == i);
    assert(i.all() == i);
    assert(i.contracted() == _);
    assert(i.projected() == _);
    assert(i.index_of('i') == 0);
    assert(i.index_of('j') == 1);
    assert(i.index_of('i', p) == 0);
    assert(p == 0);

    assert(i.count('i') == 1);
    assert(i.count('j') == 0);
    assert(i.is_subset_of(i));
    assert(ttl::is_permutation(i, i));

    ttl::index_string j { "j" };

    assert(i != j);
    assert(not i.is_subset_of(j));
    assert(not ttl::is_permutation(i, j));

    ttl::index_string ij { "ij" };

    assert(ij == i + j);

    assert(ij.size() == 2);
    assert(ij.rank() == 2);

    assert(ij.outer() == ij);
    assert(ij.inner() == ij);
    assert(ij.all() == ij);
    assert(ij.projected() == _);
    assert(ij.contracted() == _);

    assert(ij.count('i') == 1);
    assert(ij.count('j') == 1);
    assert(ij.count('k') == 0);

    assert(ij.index_of('i') == 0);
    assert(ij.index_of('j') == 1);
    assert(ij.index_of('k') == 2);

    assert(ij.index_of('i', p) == 0);
    assert(p == 0);
    assert(ij.index_of('j', p) == 1);
    assert(p == 0);

    assert(i.is_subset_of(ij));
    assert(j.is_subset_of(ij));
    assert(not ij.is_subset_of(i));
    assert(not ttl::is_permutation(i, ij));

    ttl::index_string ii = { "ii" };

    assert(ii == i + i);

    assert(ii.size() == 2);
    assert(ii.rank() == 0);
    assert(i.is_subset_of(ii));
    assert(ii.is_subset_of(i));
    assert(ttl::is_permutation(i, ii));

    assert(ii.count('i') == 2);
    
    assert(ii.outer() == _);
    assert(ii.inner() == i);
    assert(ii.all() == i);
    assert(ii.projected() == _);
    assert(ii.contracted() == i);

    ttl::index_string n { "*" };

    assert(n.size() == 1);
    assert(n.rank() == 0);
    
    assert(n.index_of('*', p) == 0);
    assert(p == 1);

    ttl::index_string inj = i + n + j;

    assert(inj.size() == 3);
    assert(inj.rank() == 2);

    assert(inj.outer() == ij);
    assert(inj.inner() == ij);
    assert(inj.all() == ij + n);
    assert(inj.contracted() == _);
    assert(inj.projected() == n);

    return true;
}

int main()
{
    constexpr auto _ = _tests();
    return 0;
}
