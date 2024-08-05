#pragma once

#include <ttl/ARROW.hpp>
#include <ttl/evaluate.hpp>
#include <ttl/expression.hpp>
#include <ttl/index.hpp>
#include <ttl/index_mapper.hpp>
#include <ttl/outer.hpp>
#include <ttl/scalar_type.hpp>
#include <ttl/tensor.hpp>
#include <concepts>

namespace ttl
{
    namespace _
    {
        template <index self, index b>
        inline constexpr auto extents_map = []<std::size_t... i>(std::index_sequence<i...>) {
            static constexpr auto map = self.index_of<b.rank()>(b);
            return std::index_sequence<map[i]...>();
        }(std::make_index_sequence<b.rank()>());
    }

    namespace tree
    {

        template <concepts::expression A, concepts::expression B>
        struct sum
        {
            static_assert(ttl::outer<A>.is_permutation_of(ttl::outer<B>));

            static constexpr auto _outer = ttl::outer<A>;
            static constexpr auto _rank = _outer.rank();
            static constexpr auto _map = _::index_mapper(_outer, ttl::outer<B>);

            A _a;
            B _b;

            static consteval auto rank() -> std::size_t {
                return _rank;
            }

            static consteval auto outer() -> decltype(_outer) {
                return _outer;
            }

            constexpr auto extents() const
                TTL_ARROW ( _::merge(_a.extents(), _b.extents()) );

            template <std::integral... I>
            requires (sizeof...(I) == _rank)
            constexpr auto operator[](this auto& self, I... i)
                TTL_ARROW ( self._a[i...] + _map(self._b, i...) );
        };

        template <concepts::expression A, concepts::expression B>
        struct product
        {
            using scalar_type = ttl::scalar_type<A, B>;

            static constexpr ttl::index _concat = ttl::outer<A> + ttl::outer<B>;
            static constexpr ttl::index _outer = _concat.exported();
            static constexpr ttl::index _inner = _outer + _concat.contracted();
            static constexpr std::size_t _rank = _outer.rank();
            static constexpr _::index_mapper _mapA = {_inner, ttl::outer<A>};
            static constexpr _::index_mapper _mapB = {_inner, ttl::outer<B>};

            A _a;
            B _b;

            static consteval auto rank() -> std::size_t {
                return _rank;
            }

            static consteval auto outer() -> decltype(_outer) {
                return _outer;
            }

            constexpr auto extents() const
                TTL_ARROW ( _::select(_::extents_map<_concat, _outer>, ttl::extents(_a)) );

            template <std::integral... Is>
            requires (sizeof...(Is) == _inner.rank())
            constexpr auto operator[](Is... i) const
                TTL_ARROW ( _mapA(_a, i...) * _mapB(_b, i...) );

            template <std::integral... Is>
            requires (_rank <= sizeof...(Is) and sizeof...(Is) < _inner.rank())
            constexpr auto operator[](Is... i) const noexcept(noexcept(operator[](i..., 0))) -> scalar_type
            {
                scalar_type sum{};
                for (auto j = 0zu, e = _extent(i...); j != e; ++j) {
                    sum += operator[](i..., j);
                }
                return sum;
            }
          private:
            constexpr auto _extent(std::integral auto... i) const {
                return extents().extent(sizeof...(i));
            }
        };
    }
}
