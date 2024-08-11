add_executable(tensor tspan.cpp)
target_link_libraries(tensor ttl::ttl)
target_compile_options(tensor PRIVATE -Wall -Werror -Wextra -pedantic)

add_executable(tspan tspan.cpp)
target_link_libraries(tspan ttl::ttl)
target_compile_options(tspan PRIVATE -Wall -Werror -Wextra -pedantic)
