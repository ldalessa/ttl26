add_executable(test traits.cpp tensor.cpp) # test.cpp
target_link_libraries(test ttl::ttl)

add_executable(index index.cpp)
target_link_libraries(index ttl::ttl)
