add_library(ttl)
target_sources(ttl 
    PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        ttl.cppm)

target_compile_features(ttl PUBLIC cxx_std_26)
target_link_libraries(ttl PUBLIC ttl::impl)

# add_executable(test_exports main.cpp)
# target_link_libraries(test_exports PRIVATE ttl)
