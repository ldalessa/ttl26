#ifndef ARROW
#define ARROW(...) decltype(__VA_ARGS__) {      \
        return __VA_ARGS__;                     \
    }
#endif
