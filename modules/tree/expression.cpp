module;

module ttl:expression;
import :extents;
import :index;
import :istring;
import :tensor;

namespace ttl::tree
{
    template <tensor, istring>
    struct bind;

    struct expression
    {
        /// Rebind an expression. This is implemented in bind.cpp in order to
        /// break the circular dependency.
        template <class T, istring... str>
        constexpr auto operator()(this T&& self, index<str>...)
            -> bind<T, (str + ...)>;

        /// Rebind an expression.
        // template <class T>
        // constexpr auto operator()(this T&& self, is_index auto... i)
        //     -> ARROW( __fwd(self)[index(i)...] );
    };
}
