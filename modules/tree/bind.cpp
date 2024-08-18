module;

module ttl:bind;
import :istring;
import :tensor;

namespace ttl::tree
{
    template <tensor A, istring _index>
    struct bind
    {
        A _a;
    };
}
