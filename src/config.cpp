//! Validated scalar, domain, and configuration values.

#include "prometheus/integrate/config.hpp"

#include "prometheus/integrate/gauss.hpp"

#include <cmath>
#include <format>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace prometheus::integrate
{

using detail::RustFloat;

//{{{ impl: Interval
std::expected<Interval, ConfigError> Interval::create(double lower, double upper)
{
    if (!std::isfinite(lower) || !std::isfinite(upper))
    {
        return std::unexpected{
            ConfigError::one(config_issue::NonFiniteInterval{.lower = lower, .upper = upper})};
    }
    if (lower >= upper)
    {
        return std::unexpected{
            ConfigError::one(config_issue::InvalidIntervalOrder{.lower = lower, .upper = upper})};
    }
    return Interval{lower, upper};
}
//}}}

//{{{ impl: Rectangle
std::expected<Rectangle, ConfigError>
Rectangle::from_bounds(double u_lower, double u_upper, double v_lower, double v_upper)
{
    auto u = Interval::create(u_lower, u_upper);
    auto v = Interval::create(v_lower, v_upper);

    // match (u, v) { (Ok, Ok) => .., (Err, Err) => concatenate, (Err(e), _) | (_, Err(e)) => e }
    if (u && v)
    {
        return Rectangle{*u, *v};
    }
    if (!u && !v)
    {
        std::vector<ConfigIssue> issues(u.error().issues().begin(), u.error().issues().end());
        issues.insert(issues.end(), v.error().issues().begin(), v.error().issues().end());
        return std::unexpected{ConfigError::from_issues(std::move(issues))};
    }
    return std::unexpected{!u ? std::move(u).error() : std::move(v).error()};
}
//}}}

//{{{ impl: Tolerance
std::expected<Tolerance, ConfigError> Tolerance::create(double absolute, double relative)
{
    if (!std::isfinite(absolute) || !std::isfinite(relative) || absolute < 0.0 || relative < 0.0 ||
        (absolute == 0.0 && relative == 0.0))
    {
        return std::unexpected{ConfigError::one(
            config_issue::InvalidTolerance{.absolute = absolute, .relative = relative})};
    }
    return Tolerance{absolute, relative};
}
//}}}

namespace detail
{

//{{{ fun: validate_subdivisions
std::expected<std::vector<double>, ConfigError> validate_subdivisions(Interval interval,
                                                                      std::vector<double> points)
{
    if (points.empty())
    {
        return points;
    }

    std::vector<ConfigIssue> issues;
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        const double value = points[index];
        if (!std::isfinite(value))
        {
            issues.emplace_back(config_issue::NonFiniteSubdivision{.index = index, .value = value});
        }
        else if (value <= interval.lower() || value >= interval.upper())
        {
            issues.emplace_back(
                config_issue::SubdivisionOutsideInterval{.index = index,
                                                         .value = value,
                                                         .lower = interval.lower(),
                                                         .upper = interval.upper()});
        }

        if (index > 0)
        {
            const double previous = points[index - 1];
            if (std::isfinite(previous) && std::isfinite(value) && value <= previous)
            {
                issues.emplace_back(
                    config_issue::SubdivisionsNotStrictlyIncreasing{.previous_index = index - 1,
                                                                    .previous = previous,
                                                                    .index = index,
                                                                    .value = value});
            }
        }
    }

    if (issues.empty())
    {
        return points;
    }
    return std::unexpected{ConfigError::from_issues(std::move(issues))};
}
//}}}

//{{{ fun: validate_axis_subdivisions
std::expected<std::pair<std::vector<double>, std::vector<double>>, ConfigError>
validate_axis_subdivisions(Rectangle domain, std::vector<double> u, std::vector<double> v)
{
    auto u_result = validate_subdivisions(domain.u(), std::move(u));
    auto v_result = validate_subdivisions(domain.v(), std::move(v));
    if (u_result && v_result)
    {
        return std::pair{std::move(*u_result), std::move(*v_result)};
    }
    if (!u_result && !v_result)
    {
        const auto u_issues = u_result.error().issues();
        const auto v_issues = v_result.error().issues();
        std::vector<ConfigIssue> issues(u_issues.begin(), u_issues.end());
        issues.insert(issues.end(), v_issues.begin(), v_issues.end());
        return std::unexpected{ConfigError::from_issues(std::move(issues))};
    }
    return std::unexpected{!u_result ? std::move(u_result).error() : std::move(v_result).error()};
}
//}}}

} // namespace detail
} // namespace prometheus::integrate

//{{{ collection: Display
namespace ti = prometheus::integrate;
namespace ci = prometheus::integrate::config_issue;
using ti::detail::RustFloat;

std::string ti::detail::display(const ti::RuleAxis& value)
{
    switch (value)
    {
    case ti::RuleAxis::OneDimensional:
        return std::format("one-dimensional");
    case ti::RuleAxis::U:
        return std::format("u-axis");
    case ti::RuleAxis::V:
        return std::format("v-axis");
    }
    std::unreachable();
}

std::string ti::detail::display(const ti::ConfigIssue& value)
{
    using ti::detail::overloaded;
    using F = ti::detail::RustFloat;
    // Rust `match self { .. }` → `std::visit` over an overload set, one lambda per variant.
    return std::visit(
        overloaded{
            [&](const ci::PolynomialDegreeOutOfRange& e)
            {
                return std::format("polynomial degree {} exceeds the representable maximum of {}",
                                   e.degree,
                                   e.maximum);
            },
            [&](const ci::PointCountOutOfRange& e)
            {
                return std::format("point count {} is outside the representable range {}..={}",
                                   e.point_count,
                                   e.minimum,
                                   e.maximum);
            },
            [&](const ci::NonFiniteInterval& e)
            {
                return std::format(
                    "interval bounds must be finite, received ({}, {})", F{e.lower}, F{e.upper});
            },
            [&](const ci::InvalidIntervalOrder& e)
            {
                return std::format("interval bounds must satisfy lower < upper, received ({}, {})",
                                   F{e.lower},
                                   F{e.upper});
            },
            [&](const ci::InvalidTolerance& e)
            {
                return std::format("tolerances must be finite and nonnegative with at least "
                                   "one positive component, received ({}, {})",
                                   F{e.absolute},
                                   F{e.relative});
            },
            [&](const ci::NonIncreasingRuleExactness& e)
            {
                return std::format("{} high-rule exactness {} must exceed low-rule exactness {}",
                                   e.axis,
                                   e.high,
                                   e.low);
            },
            [&](const ci::UnsupportedRuleDegree& e)
            {
                return std::format("{} {} rule degree {} exceeds supported maximum {}",
                                   e.axis,
                                   e.family,
                                   e.degree,
                                   e.maximum);
            },
            [&](const ci::NonFiniteSubdivision& e)
            {
                return std::format(
                    "subdivision coordinate {} must be finite, received {}", e.index, F{e.value});
            },
            [&](const ci::SubdivisionOutsideInterval& e)
            {
                return std::format(
                    "subdivision coordinate {} with value {} must be strictly inside ({}, {})",
                    e.index,
                    F{e.value},
                    F{e.lower},
                    F{e.upper});
            },
            [&](const ci::SubdivisionsNotStrictlyIncreasing& e)
            {
                return std::format("subdivision coordinates must be strictly increasing, but "
                                   "coordinate {} is {} and coordinate {} is {}",
                                   e.previous_index,
                                   F{e.previous},
                                   e.index,
                                   F{e.value});
            },
        },
        value);
}

std::string ti::detail::display(const ti::ConfigError& value)
{
    std::string out = "invalid configuration";
    for (const auto& issue : value.issues())
    {
        out += std::format("; {}", issue);
    }
    return out;
}

std::string ti::detail::display(const ti::PolynomialDegree& value)
{
    return std::format("{}", value.value());
}

std::string ti::detail::display(const ti::PointCount& value)
{
    return std::format("{}", value.value());
}

std::string ti::detail::display(const ti::Interval& value)
{
    return std::format("[{}, {}]", RustFloat{value.lower()}, RustFloat{value.upper()});
}

std::string ti::detail::display(const ti::Rectangle& value)
{
    return std::format("{} × {}", value.u(), value.v());
}

std::string ti::detail::display(const ti::Tolerance& value)
{
    return std::format("absolute {}, relative {}",
                       RustFloat{value.absolute_value()},
                       RustFloat{value.relative_value()});
}

std::string ti::detail::display(const ti::RefinementDepth& value)
{
    return std::format("{}", value.value());
}

std::string ti::detail::display(const ti::AxisDepths& value)
{
    return std::format("u {}, v {}", value.u(), value.v());
}
//}}}
