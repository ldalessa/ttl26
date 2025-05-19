#undef DNDEBUG

#include <ttl/index/istring.hpp>

static consteval bool test_istring()
{
	using ttl::index_of;
	using ttl::istring;

	constexpr istring _ = "";
	assert(_.size() == 0);
	assert(_.rank() == 0);
	assert(_.outer() == _);
	assert(_.inner() == _);
	assert(_.all() == _);

	constexpr istring i = "i";
	assert(i.size() == 1);
	assert(i.rank() == 1);
	assert(i.outer() == i);
	assert(i.inner() == i);
	assert(i.all() == i);

	constexpr istring j = "j";
	constexpr istring ij = i + j;
	assert(ij.size() == 2);
	assert(ij.rank() == 2);
	assert(ij.outer() == ij);
	assert(ij.inner() == ij);
	assert(ij.all() == ij);

	constexpr istring iji = ij + i;
	assert(iji.outer() == j);
	assert(iji.inner() == j + i);
	assert(iji.all() == j + i);

	constexpr istring p = ttl::projection;
	assert(p.size() == 1);
	assert(p.rank() == 0);

	assert(p.outer() == _);
	assert(p.inner() == _);
	assert(p.all() == p);

	constexpr istring ip = i + p;
	assert(ip.size() == 2);
	assert(ip.rank() == 1);
	assert(ip.outer() == i);
	assert(ip.inner() == i);
	assert(ip.all() == ip);

	constexpr istring pi = p + i;
	assert(pi.size() == 2);
	assert(pi.rank() == 1);
	assert(pi.outer() == i);
	assert(pi.inner() == i);
	assert(pi.all() == ip);

	constexpr istring ppijip = p + p + iji + p;
	assert(ppijip.size() == 6);
	assert(ppijip.rank() == 1);
	assert(ppijip.outer() == j);
	assert(ppijip.inner() == j + i);
	assert(ppijip.all() == j + i + p + p + p);

	constexpr std::array map_p = index_of<ppijip, ppijip.projected>;
	assert(map_p.size() == 3);
	assert(map_p[0] == 0);
	assert(map_p[1] == 1);
	assert(map_p[2] == 5);

	constexpr std::array map_i = index_of<ppijip, 'i'>;
	assert(map_i.size() == 2);
	assert(map_i[0] == 2);
	assert(map_i[1] == 4);

	constexpr std::array map_j = index_of<ppijip, u'j'>;
	assert(map_j.size() == 1);
	assert(map_j[0] == 3);

	constexpr std::array map_k = index_of<ppijip, 'k'>;
	assert(map_k.size() == 0);

	return true;
}

static_assert(test_istring());
