#pragma once
#include <ttl/concepts.hpp>
#include <ttl/tensor_traits.hpp>
#include <ttl/tensor/rank.hpp>
#include <cassert>
import std;

namespace ttl
{
	namespace stdr = std::ranges;

	/// Implement the evaluate() overload set as a function object.
	inline constexpr struct _evaluate_fn
	{
		/// Evaluate a stdlib scalar type.
		///
		/// If T deduces as an lvalue reference (either const or non-const) this
		/// will return the reference, otherwise if it's an rvalue reference it
		/// will return the value.
		template <class T>
		requires (concepts::integral<T> or concepts::floating_point<T>)
		static constexpr auto operator()(T&& t) -> T {
			return t;
		}

		/// Evaluate any type that has the tensor_trait::evaluate defined.
		template <class T, std::integral... Is>
			requires concepts::has_evaluate_trait<T, Is...>
		static constexpr auto operator()(T&& t, Is... i) ->
			ARROW( ttl::tensor_traits<std::decay_t<T>>::evaluate(FWD(t), i...) );

		/// Evaluate any forward range.
		///
		/// This simply pops a single index out of the index pack and uses it to
		/// index into the outermost range, forwarding the result to a recursive
		/// instantiation of evaluate().
		template <class T, std::integral... Is>
		requires (not concepts::has_evaluate_trait<T, std::size_t, Is...> and stdr::forward_range<T>)
		constexpr auto operator()(this auto const& self, T&& t, std::size_t i, Is... j) ->
			ARROW( self(*stdr::next(stdr::begin(t), i), j...) );

		/// Evaluate types with appropriate multidimensional index opeartors.
		///
		/// This will match mdspan, but it will also match all of the expression
		/// tree types in the tree modules.
		template <class T, std::integral... Is>
		requires (not concepts::has_evaluate_trait<T, Is...> and not stdr::forward_range<T>)
		static constexpr auto operator()(T&& t, Is... i) ->
			ARROW( FWD(t)[i...] );

		/// The _check_n functions are here to help the has_evaluate_n concept.
		///
		/// @{
		template <std::size_t N, std::size_t... i>
		constexpr auto _check_2(this auto const& self, auto&& t, std::array<std::size_t, N> const& index, std::index_sequence<i...>) ->
			ARROW( self(FWD(t), index[i]...) );

		template <std::size_t N>
		constexpr auto _check_1(this _evaluate_fn const& self, auto&& t, std::array<std::size_t, N> const& index) ->
			ARROW( self._check_2(FWD(t), index, std::make_index_sequence<N>()) );

		template <class T>
		constexpr auto _check_0(this _evaluate_fn const& self, T&& t) ->
			ARROW ( self._check_1(FWD(t), std::array<std::size_t, rank<T>>()) );
		/// @}
	} evaluate;

	/// Get the type of ttl::evaluate(T).
	template <class T>
	using evaluate_type = decltype(evaluate._check_0(FWD(std::declval<T>())));

	/// Get the scalar type for a tensor.
	///
	/// The scalar type is the value type of the result of ttl::evaluate. The
	/// default type is simply the remove_cvref_t of the result of calling
	/// evaluate, which will always work, however to reduce the compile time
	/// cost we allow either the tensor_traits or tensor class to specialize
	/// scalar_type.
	///
	/// @{
	namespace _scalar_type
	{
		template <class T>
		concept use_tensor_trait = requires {
			typename tensor_traits<std::decay_t<T>>::scalar_type;
		};

		template <class T>
		concept use_member = not use_tensor_trait<T> and requires {
			typename std::decay_t<T>::scalar_type;
		};

		template <class T>
		struct impl {
			using type = std::remove_cvref_t<evaluate_type<T>>;
		};

		template <use_tensor_trait T>
		struct impl<T> {
			using type = tensor_traits<std::decay_t<T>>::scalar_type;
		};

		template <use_member T>
		struct impl<T> {
			using type = std::decay_t<T>::scalar_type;
		};
	}

	template <class T>
	using scalar_type = _scalar_type::impl<T>::type;
	/// @}

	namespace concepts
	{
		template <class T, std::size_t N>
		concept has_evaluate_n = requires (T&& t) {
			evaluate._check_0(FWD(t));
		};
	}
}
