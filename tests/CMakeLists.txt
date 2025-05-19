add_executable(tests
  main.cpp
  bind.cpp
  cstring.cpp
  evaluate.cpp
  extents.cpp
  imap.cpp
  index.cpp
  istring.cpp
  product.cpp
  rank.cpp
  sum.cpp
  tensor.cpp
  tensor_traits.cpp
  tspan.cpp)

target_link_libraries(tests ttl)
