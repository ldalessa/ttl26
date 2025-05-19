add_executable(index index.cpp)
target_link_libraries(index ttl::ttl)
target_compile_options(index PRIVATE -Wall -Werror -Wextra -Wno-zero-length-array -pedantic)

add_executable(tensor tensor.cpp)
target_link_libraries(tensor ttl::ttl)
target_compile_options(tensor PRIVATE -Wall -Werror -Wextra -Wno-zero-length-array -pedantic)

add_executable(tspan tspan.cpp)
target_link_libraries(tspan ttl::ttl)
target_compile_options(tspan PRIVATE -Wall -Werror -Wextra -Wno-zero-length-array -pedantic)

add_executable(main main.cpp)
target_link_libraries(main ttl::ttl)
target_compile_options(main PRIVATE -Wall -Werror -Wextra -Wno-zero-length-array -pedantic)

add_executable(assign assign.cpp)
target_link_libraries(assign ttl::ttl)
target_compile_options(assign PRIVATE -Wall -Werror -Wextra -Wno-zero-length-array -pedantic)

add_executable(sum sum.cpp)
target_link_libraries(sum ttl::ttl)
target_compile_options(sum PRIVATE -Wall -Werror -Wextra -Wno-zero-length-array -pedantic)
