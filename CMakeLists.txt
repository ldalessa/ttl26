cmake_minimum_required(VERSION 3.30.2)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_FLAGS "-stdlib=libc++")
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508")

project(ttl CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_INIT} ${CXXFLAGS}")
set(CMAKE_CXX_MODULE_STD 1)

add_subdirectory(ttl)
