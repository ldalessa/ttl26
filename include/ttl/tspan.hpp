#pragma once
#include <ttl/index/index.hpp>
#include <ttl/tree/bind.hpp>
import std;

namespace ttl
{
	namespace stdr = std::ranges;

	template <
		class T,
		class Extents,
		class LayoutPolicy = std::layout_right,
		class AccessorPolicy = std::default_accessor<T>>
	struct tspan : public std::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>
	{
		/// Use all of the mdspan constructors.
		using tspan::mdspan::mdspan;

		/// Construct a tspan from an mdspan.
		constexpr tspan(std::mdspan<T, Extents, LayoutPolicy, AccessorPolicy> mdspan)
			: tspan::mdspan(std::move(mdspan))
		{
		}

		/// Construct a tspan for a contiguous range.
		/// @{
		template <class R>
			requires stdr::contiguous_range<R> and stdr::sized_range<R>
		constexpr tspan(R&& r)
			: tspan::mdspan(stdr::data(r), stdr::size(r))
		{
		}

		constexpr tspan(stdr::contiguous_range auto&& r, Extents extents)
			: tspan::mdspan(stdr::data(r), std::move(extents))
		{
		}

		constexpr tspan(stdr::contiguous_range auto&& r, std::size_t i, std::integral auto... j)
			: tspan::mdspan(stdr::data(r), Extents(i, j...))
		{
		}
		/// @}

		/// Construct a tspan for a contiguous iterator.
		/// @{
		constexpr tspan(std::contiguous_iterator auto it, Extents extents)
			: tspan::mdspan(std::to_address(it), std::move(extents))
		{
		}

		constexpr tspan(std::contiguous_iterator auto it, std::size_t i, std::integral auto... j)
			: tspan::mdspan(std::to_address(it), Extents(i, j...))
		{
		}
		/// @}

		/// Assign from a tensor.
		// template <class A>
		// constexpr auto operator=(this A&& a, tensor auto&& b) -> decltype(a) {
		//	   ttl::tree::assign(__fwd(a), __fwd(b));
		//	   return a;
		// }

		/// Tensor indexing.
		///
		/// This creates a bind node for an mdspan... tspans are never
		/// themselves bound. This allows us to write custom evaluation code for
		/// mdspan and have it work properly for things bound via the tspan.
		///
		/// Copying the mdspan is going to be cheap since it's non- owning.
		constexpr auto operator()(this auto&& self, auto... is)
			-> ARROW ( ttl::tree::bind(FWD(self), is...) );
	};

	/// Infer the scalar type and static extents for a c-array.
	///
	/// This covers both T and T const.
	template <class T, std::size_t N>
	tspan(T (&)[N]) -> tspan<T, std::extents<std::size_t, N>>;

	/// Infer the scalar type and static extents for an array.
	///
	/// Have to infer `const` indepenently.
	/// @{
	template <class T, std::size_t N>
	tspan(std::array<T, N>&) -> tspan<T, std::extents<std::size_t, N>>;

	template <class T, std::size_t N>
	tspan(std::array<T, N> const&) -> tspan<std::add_const_t<T>, std::extents<std::size_t, N>>;
	/// @}

	/// Infer the scalar type and static extents for a span.
	///
	/// This covers both T and T const.
	template <class T, std::size_t N>
	tspan(std::span<T, N>) -> tspan<T, std::extents<std::size_t, N>>;

	/// Infer the scalar type and dynamic extents for a contiguous range.
	template <class R>
		requires stdr::contiguous_range<R> and stdr::sized_range<R>
	tspan(R&&)
		-> tspan<
			std::remove_reference_t<stdr::range_reference_t<R>>,
			std::extents<std::size_t, std::dynamic_extent>>;

	/// Infer T as the range value type.
	template <stdr::contiguous_range R, class T, std::size_t... Es>
	tspan(R&&, std::extents<T, Es...>)
		-> tspan<
			std::remove_reference_t<stdr::range_reference_t<R>>,
			std::extents<T, Es...>>;

	/// Infer T as the range value type, and the dynamic extents.
	template <stdr::contiguous_range R>
	tspan(R&&, std::integral auto, std::integral auto... i)
		-> tspan<
			std::remove_reference_t<stdr::range_reference_t<R>>,
			std::extents<std::size_t, std::dynamic_extent, ((void)i, std::dynamic_extent)...>>;

	/// Infer T as the iterator value type.
	template <std::contiguous_iterator It, class T, std::size_t... Es>
	tspan(It const&, std::extents<T, Es...>)
		-> tspan<
			std::remove_reference_t<std::iter_reference_t<It>>,
			std::extents<T, Es...>>;

	/// Infer T as the iterator value type, and the dynamic extents.
	template <std::contiguous_iterator It, std::integral... I>
	tspan(It const&, std::integral auto... i)
		-> tspan<
			std::remove_reference_t<std::iter_reference_t<It>>,
			std::extents<std::size_t, ((void)i, std::dynamic_extent)...>>;
}
