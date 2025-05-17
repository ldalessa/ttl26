export module ttl:expression;
export import :extents;
export import :evaluate;
export import :index;
export import :istring;
export import :tensor;
import std;

namespace ttl::tree
{
	/// Forward declare the bind tree node type, and provide the type deduction
	/// for it. This lets us hoist rebinding into the expression base class. The
	/// :bind module imports :expression in order to inherit from the expression
	/// type, so these deduction guides are available there.
	///
	/// expression::_rebind is implemented in bind.cpp, once the bind
	/// constructor is actually available.
	///
	/// @{
	template <tensor, istring>
	struct bind;

	template <class A, istring... is>
	bind(A&, index<is>...) -> bind<A&, (istring{""} + ... + is)>;

	template <class A, istring... is>
	bind(A const&&, index<is>...) -> bind<A const, (istring{""} + ... + is)>;

	template <class A, class I, class... Is>
		requires (std::integral<I> or ... or std::integral<Is>)
	bind(A&, I, Is...) -> bind<A&, (to_istring<I> + ... + to_istring<Is>)>;

	template <class A, class I, class... Is>
		requires (std::integral<I> or ... or std::integral<Is>)
	bind(A const&&, I, Is...) -> bind<A const, (to_istring<I> + ... + to_istring<Is>)>;
	/// @}

	struct expression
	{
		/// Neither copyable nor movable, for now.
		constexpr expression() = default;
		constexpr expression(expression const&) = delete;
		constexpr expression(expression&&) = delete;
		constexpr auto operator=(expression const&) -> expression& = delete;
		constexpr auto operator=(expression&&) -> expression& = delete;
		
		/// Indexing should use operator[]
		constexpr auto operator()(this auto&&, std::integral auto...) = delete;

		/// Rebind an expression.
		constexpr auto operator()(this auto&& self, auto... is)
			-> ARROW( FWD(self)._rebind(index(is)...) );

		/// Allow scalar expressions to decay to their scalar value.
		template <concepts::scalar T>
		constexpr operator evaluate_type<T>(this T&& self) {
			return FWD(self)[];
		}

		/// Allow scalar expressions to be assigned from an appropriate scalar
		/// type.
		template <concepts::scalar T, concepts::scalar U>
		requires (not std::same_as<std::remove_cvref_t<T>, std::remove_cvref_t<U>>)
		constexpr auto operator=(this T& self, U&& u)
			-> ARROW( FWD(self)[] = FWD(u) );

	  protected:
		/// Rebind an expression.
		///
		/// This is implemented in :bind in order to break the circular
		/// dependency.
		///
		/// @note The expression type T is needed in the implementation to
		///		  verify that the rebound indices cover the current indices, so
		///		  it has to be declared in this form rather than using an
		///		  abbreviated form. This is obvious when you go look at the
		///		  implementation in bind.cppm, even if it's not obvious here.
		///
		template <concepts::expression T, istring... str>
		constexpr auto _rebind(this T&& self, index<str>... is)
			-> decltype(bind(FWD(self), is...));

		/// Check that contracted extents are statically compatible.
		///
		/// Contracted extents i and j are	"statically compatible" if either
		/// static_extent is std::dynamic range, or if the extents are equal.
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
