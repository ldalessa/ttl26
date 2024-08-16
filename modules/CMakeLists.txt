add_library(ttl)
target_sources(ttl 
    PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        ttl.cpp
        concepts.cpp
        cstring.cpp
        extents.cpp
        evaluate.cpp
        index.cpp
        istring.cpp
        rank.cpp
        tensor_traits.cpp
        tspan.cpp)

target_compile_options(ttl PUBLIC
  --include=${CMAKE_CURRENT_SOURCE_DIR}/FWD.hpp
  --include=${CMAKE_CURRENT_SOURCE_DIR}/ARROW.hpp)

target_compile_features(ttl PUBLIC cxx_std_26)

# add_executable(demo main.cpp)
# target_link_libraries(demo PRIVATE ttl)
