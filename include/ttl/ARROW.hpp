#ifndef TTL_ARROW
#define TTL_ARROW(...)                          \
    noexcept(noexcept(__VA_ARGS__))             \
    -> decltype(__VA_ARGS__)                    \
    {                                           \
        return __VA_ARGS__;                     \
    }
#endif
