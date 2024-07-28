#pragma once

#include <concepts>
#include <format>

namespace ttl
{
    namespace concepts
    {
        template <class T>
        concept formattable = requires (T const& t, std::format_context& ctx) {
            { t.format(ctx) } -> std::convertible_to<decltype(ctx.out())>;
        };
    }
}

template <ttl::concepts::formattable T>
struct std::formatter<T>
{
    static constexpr auto parse(std::format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    // clang-18 won't let me say std::format_context here
    static constexpr auto format(T const& t, auto& ctx) -> decltype(t.format(ctx)) {
        return t.format(ctx);
    }
};
