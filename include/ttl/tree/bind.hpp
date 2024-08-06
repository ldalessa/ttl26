#pragma once

#include <ttl/ARROW.hpp>
#include <ttl/concepts.hpp>
#include <ttl/sequence.hpp>
#include <ttl/tensor.hpp>
#include <ttl/tensor_extents.hpp>
#include <ttl/tensor_index.hpp>
#include <ttl/tree/_extents_map.hpp>
#include <ttl/tree/_index_mapper.hpp>

#include <concepts>
#include <cstddef>
#include <utility>

namespace ttl
{
    namespace tree
    {
        template <tensor_index _index, concepts::tensor A>
        struct bind
        {
            friend struct ttl::traits<bind>;

            using scalar_type = ttl::scalar_type<A>;

            static constexpr auto _outer = _index.exported();
            static constexpr auto _inner = _outer + _index.contracted();
            static constexpr auto _rank = _outer.rank();
            static constexpr auto _map = _index_mapper<_index.size()>(_inner, _index);
            static constexpr auto _outer_extents_map = _extents_map<_index, _outer>;
            static constexpr auto _inner_extents_map = _extents_map<_index, _inner>;

            A _a;

            constexpr bind(A a) : _a(a) {
                assert(_check_contracted_extents() && "contracted extents have different sizes");
            }

            /// The index operator forwards to _evaluate.
            ///
            /// Forwarding is necessary because, when the bind index contains a
            /// contraction, we can wind up in a recursive-ish accumulation loop
            /// and it's much simpler to do that internally in _evaluate than
            /// externally via recursive-ish operator[].
            template <class T>
            constexpr auto operator[](this T&& self, concepts::size_t auto... i)
                TTL_ARROW ( std::forward<T>(self)._evaluate(std::size_t(i)...) );

            /// If the bind represents a scalar we can implicitly convert to the
            /// scalar_type.
            constexpr operator scalar_type() const requires (_rank == 0) {
                return _evaluate();
            }

          private:
            ///
            constexpr auto _outer_extents() const
                TTL_ARROW ( ttl::select_extents(_outer_extents_map, ttl::tensor_extents(_a)) );

            constexpr auto _outer_extent(concepts::size_t auto... i) const {
                static_assert(_rank <= sizeof...(i));
                return _outer_extents().extent(sizeof...(i));
            }

            ///
            constexpr auto _inner_extents() const
                TTL_ARROW ( ttl::select_extents(_inner_extents_map, ttl::tensor_extents(_a)) );

            constexpr auto _inner_extent(concepts::size_t auto... i) const {
                static_assert(_rank <= sizeof...(i));
                return _inner_extents().extent(sizeof...(i));
            }

            /// This is the innermost _evaluate function. It will map the passed
            /// indices into the underlying space (this might require reordering
            /// and duplication when this bind represents a trace).
            template <class T, concepts::size_t... Is>
            requires (sizeof...(Is) == _inner.rank())
            constexpr auto _evaluate(this T&& self, Is... i)
                TTL_ARROW ( _map(std::forward<T>(self)._a, std::size_t(i)...) );

            /// This version of _evaluate will get called if there is a
            /// contraction in this bind (basically it's going to be a
            /// trace). It will recursively call itself for each contracted
            /// extent..
            constexpr auto _evaluate(concepts::size_t auto... i) const noexcept(noexcept(_evaluate(std::size_t(i)..., 0zu))) -> scalar_type
                requires (_rank <= sizeof...(i) and sizeof...(i) < _inner.rank())
            {
                static_assert(_rank <= sizeof...(i) and sizeof...(i) < _inner.rank());
                scalar_type sum{};
                for (auto j = 0zu, e = _inner_extent(i...); j != e; ++j) {
                    sum += _evaluate(std::size_t(i)..., j);
                }
                return sum;
            }

            constexpr auto _check_contracted_extents() const -> bool
            {
                using Extents = decltype(auto(ttl::tensor_extents(_a)));

                static constexpr auto contracted = _index.contracted();
                static constexpr auto N = contracted.size();

                static constexpr auto find_offsets = [](char const c) -> std::array<int, 2> {
                    std::array<int, 2> out;
                    for (int i = 0, n = 0; auto const d : _index) {
                        if (c == d) out[n++] = i;
                        i += 1;
                    }
                    return out;
                };

                auto const a_extents = ttl::tensor_extents(_a);

                return [&]<std::size_t... i>(sequence<i...>) {
                    return ([&] {
                        constexpr auto x = find_offsets(contracted[i]);
                        assert(x[0] != x[1]);
                        static_assert(Extents::static_extent(x[0]) == Extents::static_extent(x[1]) or
                                      Extents::static_extent(x[0]) == std::dynamic_extent or
                                      Extents::static_extent(x[1]) == std::dynamic_extent);
                        return a_extents.extent(x[0]) == a_extents.extent(x[1]);
                    }() && ...);
                }(seqn<N>);
            }
        };
    }

    template <tensor_index _index, concepts::tensor A>
    struct traits<tree::bind<_index, A>>
    {
        using bind = tree::bind<_index, A>;

        static consteval auto outer() {
            return bind::_outer;
        }

        static constexpr auto tensor_extents(bind const& x)
            TTL_ARROW ( x._outer_extents() );

        static constexpr auto evaluate(bind const& x, concepts::size_t auto... i)
            TTL_ARROW ( x._evaluate(std::size_t(i)...) );

        static constexpr auto evaluate(bind& x, concepts::size_t auto... i)
            TTL_ARROW ( x._evaluate(std::size_t(i)...) );
    };

    template <tensor_index _index, concepts::tensor A>
    inline constexpr auto bind(A&& a) -> tree::bind<_index, A> {
        // return tree::bind<_index, A>{ ._a = std::forward<A>(a) };
        return tree::bind<_index, A>{ std::forward<A>(a) };

    }
}
