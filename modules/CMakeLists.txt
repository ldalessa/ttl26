add_library(ttl)
target_sources(ttl 
    PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        ttl.cpp
        concepts.cpp
        tensor_traits.cpp
        tspan.cpp
        index/cstring.cpp
        index/imap.cpp
        index/index.cpp
        index/istring.cpp
        tensor/extents.cpp
        tensor/evaluate.cpp
        tensor/outer.cpp
        tensor/rank.cpp
        tensor/tensor.cpp
        tree/bind.cpp
        tree/expression.cpp)

target_compile_options(ttl PUBLIC
  --include=${CMAKE_CURRENT_SOURCE_DIR}/FWD.hpp
  --include=${CMAKE_CURRENT_SOURCE_DIR}/ARROW.hpp)

target_compile_features(ttl PUBLIC cxx_std_26)

add_executable(test_exports main.cpp)
target_link_libraries(test_exports PRIVATE ttl)
