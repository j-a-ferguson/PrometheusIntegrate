//! Rust-compatible floating-point `Display`.

#include "prometheus/integrate/detail/display.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <string>
#include <system_error>

namespace prometheus::integrate::detail
{

std::string rust_display(double value)
{
    if (std::isnan(value))
    {
        return "NaN";
    }
    if (std::isinf(value))
    {
        return value > 0.0 ? "inf" : "-inf";
    }
    // `std::to_chars` with an explicit `fixed` format and no precision yields the *shortest*
    // digit string that round-trips, written without an exponent — exactly Rust's `{}` for f64
    // (e.g. 1e-8 → "0.00000001", 2.0 → "2"). The largest finite double needs 309 digits.
    std::array<char, 400> buffer{};
    const auto [end, error] = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::fixed);
    if (error != std::errc{})
    {
        return std::to_string(value);
    }
    return {buffer.data(), end};
}

} // namespace prometheus::integrate::detail
