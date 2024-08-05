add_executable(test test.cpp #index.cpp traits.cpp tensor.cpp
  bind.cpp)
target_link_libraries(test ttl::ttl)
