module;

#include <cstddef>
#include <mdspan>
#include <utility>

module ttl:expression;
import :extents;
import :evaluate;
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
        // template <class T, istring... str>
        // constexpr auto operator()(this T&& self, index<str>...)
        //     -> bind<T, (str + ...)>;

        /// Rebind an expression.
        // template <class T>
        // constexpr auto operator()(this T&& self, is_index auto... i)
        //     -> ARROW( __fwd(self)[index(i)...] );

        /// Allow scalar expressions to decay to their scalar value.
        ///
        /// @todo This only currently (clang 18.1, gcc 14.2) works for const
        /// uses, because compilers disagree about what it should mean:
        /// https://godbolt.org/z/xWMKzna6W.
        template <class T>
        constexpr operator evaluate_type<T>(this T&& self) {
            return FWD(self)[];
        }

      protected:
        template <istring index, class Extents>
        static constexpr bool _check_contracted_extents_static = [] {
            for (auto const c : index.contracted()) {
                auto [i, j] = index.index_of_2(c);
                auto a = Extents::static_extent(i);
                auto b = Extents::static_extent(j);
                if (a != std::dynamic_extent and b != std::dynamic_extent and a != b) {
                    return false;
                }
            }
            return true;
        }();

        /// Check that contracted indices have the same extents.
        template <istring index, std::size_t... es>
        static constexpr bool _check_contracted_extents_dynamic(std::extents<std::size_t, es...> const& extents)
        {
            for (auto const c : index.contracted()) {
                auto [i, j] = index.index_of_2(c);
                if (extents.extent(i) != extents.extent(j)) {
                    return false;
                }
            }
            return true;
        }
    };
}
