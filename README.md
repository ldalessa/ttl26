# ttl26
A modern, c++26-inspired rewrite of my tensor template library.

# Design

TTL is a tensor algebra expression-template domain specific language in the
tradition of Blaze, Blitz++, Eigen, etc. It shares the funamental idea of
encoding algebraic expressions as a set of recursive template types, for which
optimized code can be emitted by the library using a variety of template
metaprogramming techniques.

TTL differs from prior work in three ways.

1. It is designed from the ground up around tensors and tensor algebra, rather
than linear algebra. In service of this design goal, TTL models Einstein
notation, where tensor expressions are annotated with explicit tensor indices
that correspond to implied summation.

2. Its interface is entirely designed around concepts and non-owning
spans/multi-dimensional spans, and it does not provide any owning containers.

3. It is designed and written to use the most modern C++ features that are
available in the most recently released compilers (C++26 at the time of this
writing).

# API Components.

## `concept ttl::tensor`

The `tensor` concept requires that a type support two functions, `ttl::extents`
and `ttl::evaluate`. The `extents` function must return a
`std::extents<std::size_t, ...>` object that denotates the shape of the tensor,
and the `evaluate` function must take a variadic pack of integers of size equal
to the rank of the tensor and return the scalar at that index.

The library provides `tensor` support for standard library types inculding
`std::integral`, `std::floating_point`, `std::is_bounded_array`, `std::array`,
`std::vector`, `std::span`, and `std::mdspan`. It also provides support for any
type that satisfies `std::ranges::range`.

User types that are not `std::ranges::range` are supported through specializing
a customization point trait called `ttl::tensor_traits`. Even if a type
satisfies `std::ranges::range`, it may be more efficient to specialize the
traits.

## `template struct ttl::index`

The `index` template class represents an explicit index. Individual indices take
the form of a single `u''` character, such as `'i', 'j', u'μ', and must be set
statically at compile time in the binary. Compile-time indices may be specified
to the `ttl::index` template class either via a non-type template parameter or
using the index literal. Compound indices can be specified directly or via
`operator+` concatentation.

```c++
ttl::index<"i"> i;

using namespace ttl::literals;
auto j = "j"_i;`

ttl::index ij = i + j; // concatenate
ttl::index<"kl"> kl;
```

## `ttll::bind`

"Binding" is the act of taking a tensor and annotating it with an index so that
it can be used in an expression. TTL can automatically bind things that it knows
are scalars so they can be used directly in expression. TTL can also infer bound
indices for binary operations where one subexpression is bound.

```c++
int a_data[9]{};
int x_data[3]{};
int y_data[3]{};

auto A = std::mdspan(a_data, 3, 3);
auto x = std::mdspan(x_data, 3);
auto y = std::mdspan(y_data, 3);

// Explicitly bind all three tensors.
ttl::bind(y, i) = ttl::bind(A, i, j) * ttl::bind(x, j);

// Scalars don't need to be bound
ttl::bind(y, i) = M_PI * ttl::bind(A, i, j) * ttl::bind(x, j);

// If one part of an expression is bound, the rest of the expression can be
// inferred as long as ranks match.
y = M_PI * ttl::bind(A, i, j) * ttl::bind(x, j);
y_data = M_PI * ttl::bind(A, i, j) * ttl::bind(x, j);
```

## `ttl::tspan`

TTL does not provide any owning containers, but it does provide a "tensor span"
which is morally equivalent to a `std::mdspan` but provides an `operator()` that
can be used directly in an expression.

```c++
int a_data[9]{};
int x_data[3]{};
int y_data[3]{};

auto A = ttl::tspan(a_data, 3, 3);
auto x = ttl::tspan(x_data, 3);
auto y = ttl::tspan(y_data, 3);

y = A(i,j) * x(j);
```

The tensor span provides constructors from standard `ranges`, `span`, and
`mdspan`, as well as the same binding of one-dimensional storage that `mdspan`
provides.

## Expressions

There is no specific type for expressions, they are simply the result of binding
tensors and the corresponding operators. While not exactly part of the public
API, each operator will produce some sort of composite tree node from the
`ttl::tree` namespace which you will encounter in compiler error messages. See
the `/ttl/tree` files for specific details.

The two core expression types are tensor sums and tensor products. Sums are
binary operators that might either be plus or minus, and are performed
element-wise with the potential for transposing based on the tensor indices
involved. Products are tensor products where the common indices between the
right and left hand side are summed along their extents. Expressions may also
include projections which are specified using an integer index in one slot
rather than a tensor index.

```c++
ttl::index<"i"> i;
ttl::index<"j"> j;
ttl::index<"k"> k;
ttl::index<"l"> l;

ttl::tspan A(...), B(...), C(...), x(...), y(...), z(...);

z(i) = x(i) + y(i);
C(i,j) = A(i,j) + B(i,j);
C(i,j) = A(i,j) + B(j,i); // "transposes" B
z(i) = x(i) + B(2,i); // proectes an extent of B

z(i) = 2 * x(i); // scalar product
z(i) = A(i,j) * x(j); // "normal" matrix-vector product
z(i) = x(j) * A(i,j); // "normal" matrix-vector produce (commutative)
C(i,j) = x(i) * y(j);; // outer product
z(j) = A(i,j) * x(i); // transposed matrix-vector product
C(i,j) = A(i,k) * B(k, j); // "normal" matrix-matrix product
int trace = A(i,i); // trace

ttl::tspan D(...);
D(i,j,k,l) = A(i,j) * B(k,l); // outer product
```

## Library types and functions

There are convenience types in the library like the delta function and
functions. 

## Aliasing

## In-Tree Temporaries

## Expression pattern recognition and offloading
