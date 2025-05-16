add_library(ttl)
target_sources(ttl 
    PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        ttl.cppm
        concepts.cppm
        tensor_traits.cppm
        tspan.cppm
        index/cstring.cppm
        index/imap.cppm
        index/index.cppm
        index/istring.cppm
        tensor/extents.cppm
        tensor/evaluate.cppm
        tensor/outer.cppm
        tensor/rank.cppm
        tensor/tensor.cppm
        tree/bind.cppm
        tree/expression.cppm)

target_compile_options(ttl PUBLIC
  --include=${CMAKE_CURRENT_SOURCE_DIR}/FWD.hpp
  --include=${CMAKE_CURRENT_SOURCE_DIR}/ARROW.hpp)

target_compile_features(ttl PUBLIC cxx_std_26)

add_executable(test_exports main.cpp)
target_link_libraries(test_exports PRIVATE ttl)
