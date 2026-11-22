#pragma once
//! Results and errors produced while evaluating integrals.

//{{{ crate includes
#include "prometheus/integrate/config.hpp"
#include "prometheus/integrate/detail/display.hpp"
//}}}
//{{{ std includes
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <variant>
//}}}

namespace prometheus::integrate
{

//{{{ concepts: Integrand1d / Integrand2d
/// Rust `F: FnMut(f64) -> f64`. Integrands are always invoked as lvalues, so mutable lambdas
/// and stateful function objects may update their state between evaluations.
template <class F>
concept Integrand1d =
    std::invocable<F&, double> && std::convertible_to<std::invoke_result_t<F&, double>, double>;

/// Rust `F: FnMut(f64, f64) -> f64`.
template <class F>
concept Integrand2d = std::invocable<F&, double, double> &&
                      std::convertible_to<std::invoke_result_t<F&, double, double>, double>;
//}}}

//{{{ variant: EvaluationPoint
namespace evaluation_point
{

/// A one-dimensional coordinate.
struct OneDimensional
{
    double x;
    bool operator==(const OneDimensional&) const = default;
};

/// A two-dimensional coordinate.
struct TwoDimensional
{
    double u;
    double v;
    bool operator==(const TwoDimensional&) const = default;
};

} // namespace evaluation_point

/// Coordinates at which an integrand was evaluated.
using EvaluationPoint =
    std::variant<evaluation_point::OneDimensional, evaluation_point::TwoDimensional>;
//}}}

//{{{ class: AdaptiveResult
/// Value and diagnostics produced by adaptive integration.
class AdaptiveResult
{
public:
    constexpr AdaptiveResult(double integral,
                             double error_estimate,
                             std::size_t terminal_region_count,
                             std::size_t evaluation_count) noexcept
        : integral_{integral}, error_estimate_{error_estimate},
          terminal_region_count_{terminal_region_count}, evaluation_count_{evaluation_count}
    {
    }

    /// Returns the high-order approximation to the integral.
    [[nodiscard]] constexpr double integral() const noexcept { return integral_; }
    /// Returns the sum of the terminal regions' low/high differences.
    [[nodiscard]] constexpr double error_estimate() const noexcept { return error_estimate_; }
    /// Returns the number of terminal regions.
    [[nodiscard]] constexpr std::size_t terminal_region_count() const noexcept
    {
        return terminal_region_count_;
    }
    /// Returns the total number of integrand evaluations.
    [[nodiscard]] constexpr std::size_t evaluation_count() const noexcept
    {
        return evaluation_count_;
    }

    bool operator==(const AdaptiveResult&) const = default;

private:
    double integral_;
    double error_estimate_;
    std::size_t terminal_region_count_;
    std::size_t evaluation_count_;
};
//}}}

//{{{ variant: IntegrationError
namespace integration_error
{

/// The integrand returned NaN or infinity.
struct NonFiniteIntegrand
{
    EvaluationPoint point;
    double value;
    bool operator==(const NonFiniteIntegrand&) const = default;
};

/// The requested global tolerance could not be met within the configured depth limits.
struct MaxDepthReached
{
    AdaptiveResult partial_result;
    bool operator==(const MaxDepthReached&) const = default;
};

/// A one-dimensional region was too narrow to bisect in floating-point arithmetic.
struct NonProgressingInterval
{
    Interval interval;
    AdaptiveResult partial_result;
    bool operator==(const NonProgressingInterval&) const = default;
};

/// A two-dimensional region was too narrow to split in floating-point arithmetic.
struct NonProgressingRectangle
{
    Rectangle rectangle;
    AdaptiveResult partial_result;
    bool operator==(const NonProgressingRectangle&) const = default;
};

} // namespace integration_error

/// An error encountered while evaluating an integral.
using IntegrationError = std::variant<integration_error::NonFiniteIntegrand,
                                      integration_error::MaxDepthReached,
                                      integration_error::NonProgressingInterval,
                                      integration_error::NonProgressingRectangle>;
//}}}

} // namespace prometheus::integrate

PROMETHEUS_INTEGRATE_DECLARE_FORMATTER(prometheus::integrate::EvaluationPoint);
PROMETHEUS_INTEGRATE_DECLARE_FORMATTER(prometheus::integrate::IntegrationError);
