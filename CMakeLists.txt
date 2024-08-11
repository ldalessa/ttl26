cmake_minimum_required(VERSION 3.30.2)
project(ttl CXX)

option(TTL_ENABLE_TESTS "Build the tests diretory." ON)

add_library(ttl_lib INTERFACE)
target_include_directories(ttl_lib INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_compile_features(ttl_lib INTERFACE cxx_std_26)
target_compile_options(ttl_lib INTERFACE -include ttl/__fwd.hpp)
add_library(ttl::ttl ALIAS ttl_lib)

if (TTL_ENABLE_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif (TTL_ENABLE_TESTS)
