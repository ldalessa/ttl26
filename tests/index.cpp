#undef NDEBUG

#include <ttl/index_string.hpp>

#include <cassert>

static constexpr bool _constructor()
{
    ttl::index_string _{};
    ttl::index_string _{"i"};
    ttl::index_string _{"j"};
    ttl::index_string _{"ij"};
    ttl::index_string _{"*"};

    return true;
}

int main()
{
    assert(_constructor());
    return 0;
}
