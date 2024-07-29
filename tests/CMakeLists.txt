add_executable(test concepts.cpp) # test.cpp tensor.cpp)
target_link_libraries(test ttl::ttl)

add_executable(index index.cpp)
target_link_libraries(index ttl::ttl)
