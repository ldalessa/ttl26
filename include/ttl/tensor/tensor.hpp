#pragma once
#include <ttl/tensor/evaluate.hpp>
#include <ttl/tensor/extents.hpp>
#include <ttl/tensor/outer.hpp>
#include <ttl/tensor/rank.hpp>
import std;

namespace ttl::concepts
{
	template <class T>
	concept tensor = has_extents<T> and has_evaluate_n<T, rank<T>>;

	template <class T, std::size_t N>
	concept tensor_of_rank = tensor<T> and rank<T> == N;
	
	template <class T>
	concept expression = tensor<T> and has_outer<T>;

	template <class T>
	concept scalar = expression<T> and tensor_of_rank<T, 0>;

}
