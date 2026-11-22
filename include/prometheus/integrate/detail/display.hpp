#pragma once
//! Shared plumbing for the `std::formatter` specializations that replace Rust's `Display` impls.

//{{{ std includes
#include <algorithm>
#include <concepts>
#include <format>
#include <string>
#include <type_traits>
#include <utility>
//}}}

namespace prometheus::integrate
{

//{{{ concept: Displayable
/// Rust `T: Display` → a type with an enabled `std::formatter<T>` specialization.
///
/// Disabled `std::formatter` specializations are not default-constructible, which is the same
/// test the standard's `std::formattable` concept builds on.
template <class T>
concept Displayable = std::semiregular<std::formatter<std::remove_cvref_t<T>, char>>;
//}}}

//{{{ fun: to_string
/// Rust `value.to_string()` for any type that implements `Display`.
template <Displayable T>
[[nodiscard]] std::string to_string(const T& value)
{
    return std::format("{}", value);
}
//}}}

namespace detail
{

//{{{ struct: overloaded
/// The usual "overload set from lambdas" helper used with `std::visit`, i.e. the arms of a
/// Rust `match`.
template <class... Ts>
struct overloaded : Ts...
{
    using Ts::operator()...;
};
//}}}

//{{{ struct: NoSpecFormatter
/// Base for formatters that accept only an empty format spec (`{}`), as Rust `Display` does.
struct NoSpecFormatter
{
    constexpr auto parse(std::format_parse_context& ctx) -> std::format_parse_context::iterator
    {
        auto it = ctx.begin();
        if (it != ctx.end() && *it != '}')
        {
            throw std::format_error("PrometheusIntegrate types accept no format specifiers");
        }
        return it;
    }
};
//}}}

//{{{ struct: RustFloat
/// Wrapper that formats an `f64` exactly as Rust's `Display` does: shortest round-trip digits,
/// never in exponent notation, `NaN`, `inf` and `-inf` for non-finite values.
struct RustFloat
{
    double value;
};

[[nodiscard]] std::string rust_display(double value);
//}}}

} // namespace detail
} // namespace prometheus::integrate

/// Declares `detail::display(const Type&)` (defined out of line in a `.cpp`, keeping the message
/// text of Rust's `#[error(...)]` / `Display` bodies out of the headers) and a `std::formatter`
/// that forwards to it.
///
/// `format` must be a template over the format context: the standard's *Formatter* requirements
/// demand it work with any `basic_format_context<Out, char>`, and libc++'s compile-time format
/// string checker instantiates it with `Out = char*`.
#define PROMETHEUS_INTEGRATE_DECLARE_FORMATTER(Type)                                               \
    namespace prometheus::integrate::detail                                                        \
    {                                                                                              \
    [[nodiscard]] std::string display(const Type& value);                                          \
    }                                                                                              \
    template <>                                                                                    \
    struct std::formatter<Type> : prometheus::integrate::detail::NoSpecFormatter                   \
    {                                                                                              \
        template <class FormatContext>                                                             \
        auto format(const Type& value, FormatContext& ctx) const                                   \
        {                                                                                          \
            const std::string text = prometheus::integrate::detail::display(value);                \
            return std::copy(text.begin(), text.end(), ctx.out());                                 \
        }                                                                                          \
    }

template <>
struct std::formatter<prometheus::integrate::detail::RustFloat>
    : prometheus::integrate::detail::NoSpecFormatter
{
    template <class FormatContext>
    auto format(prometheus::integrate::detail::RustFloat value, FormatContext& ctx) const
    {
        const auto text = prometheus::integrate::detail::rust_display(value.value);
        return std::copy(text.begin(), text.end(), ctx.out());
    }
};
