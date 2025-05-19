add_executable(tests
  main.cpp
  add.cpp
  bind.cpp
  cstring.cpp
  evaluate.cpp
  extents.cpp
  imap.cpp
  index.cpp
  istring.cpp
  multiply.cpp
  rank.cpp
  subtract.cpp
  tensor.cpp
  tensor_traits.cpp
  tspan.cpp)

target_link_libraries(tests ttl)
