#include <ttl/tensor_index.hpp>
#include <ttl/formattable.hpp>

constexpr ttl::tensor_index _;
constexpr ttl::tensor_index i = "i";
constexpr ttl::tensor_index j{"j"};
constexpr ttl::tensor_index ij = i + j;
constexpr ttl::tensor_index ij2 = ij + ij;

static_assert(i == i);
static_assert(i != j);
static_assert(ij == ij);
static_assert(ij2 != ij);

static_assert(_.is_subset_of(ij));
static_assert(i.is_subset_of(ij));
static_assert(ij.is_subset_of(ij));
static_assert(not ij.is_subset_of(_));
static_assert(not ij.is_subset_of(i));

static_assert(not _.is_permutation_of(ij));
static_assert(not i.is_permutation_of(ij));
static_assert(ij.is_permutation_of(ij));
static_assert(ij.is_permutation_of(j + i));

static_assert(i[0] == 'i');
static_assert(j[0] == 'j');
static_assert(ij[0] == 'i');
static_assert(ij[1] == 'j');

static_assert(i.count('i') == 1);
static_assert(i.count('j') == 0);

static_assert(j.count('i') == 0);
static_assert(j.count('j') == 1);

static_assert(ij.count('i') == 1);
static_assert(ij.count('j') == 1);

static_assert(ij2.count('i') == 2);
static_assert(ij2.count('j') == 2);

static_assert(ij2.exported() == _);
static_assert(ij2.contracted() == ij);

static_assert((i + i + j).exported() == j);
static_assert((i + i + j).contracted() == i);

#include <print>

int main()
{
    std::print("{}\n", ij2);
}
