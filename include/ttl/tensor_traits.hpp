#pragma once
#include <ttl/concepts.hpp>
import std;

namespace ttl
{
	/// Tensor traits allow 3rd party types to be used as tensors.
	template <class>
	struct tensor_traits;
	/// {
	///		@required
	///		static contexpr auto extents(T&&) -> concepts::extents;
	///
	///		@required
	///		static contsexpr auto evaluate(T&&, std::integral auto...) -> scalar(&)
	///
	///		@optional
	///		static constexpr auto rank() -> std::convertible_to<std::size_t>
	///
	///		@optional
	///		using extents_type = ...;
	///
	///		@optional
	///		using scalar_type = ...;
	/// };
	namespace concepts
	{
		template <class T>
		concept has_extents_trait = requires (T&& t) {
			{ tensor_traits<std::decay_t<T>>::extents(FWD(t)) } -> extents;
		};

		template <class T, class... I>
		concept has_evaluate_trait = requires (T&& t, I... i) {
			tensor_traits<std::decay_t<T>>::evaluate(FWD(t), i...);
		};

		template <class T>
		concept has_rank_trait = requires {
			{ tensor_traits<std::decay_t<T>>::rank } -> concepts::integral_constant;
		};
	}
}
