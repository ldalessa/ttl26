cmake_minimum_required(VERSION 4.0.2)

set(CMAKE_CXX_FLAGS "-stdlib=libc++")
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "a9e1cf81-9932-4810-974b-6eccaf14e457")
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_MODULE_STD 1)
project(ttl CXX)
unset(CMAKE_CXX_FLAGS)

add_library(ttl_impl INTERFACE)
target_include_directories(ttl_impl INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_compile_features(ttl_impl INTERFACE cxx_std_26)
target_compile_options(ttl_impl INTERFACE
  --include=ttl/FWD.hpp
  --include=ttl/ARROW.hpp)

add_library(ttl::impl ALIAS ttl_impl)

add_subdirectory(ttl)
add_subdirectory(tests)
