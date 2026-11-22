#pragma once
//! Validated scalar, domain, and configuration values.
//!
//! Rust → C++ mapping used throughout this file:
//!   * `Result<T, ConfigError>`  → `std::expected<T, ConfigError>`
//!   * `Type::new(..)`           → `Type::create(..)` (`new` is a C++ keyword)
//!   * `pub(crate) new_unchecked`→ public `new_unchecked` (C++ has no crate-private visibility)
//!   * data-carrying `enum`      → `std::variant` of plain structs in a namespace named after it
//!   * unit-only `enum`          → `enum class`
//!   * `Display`                 → `std::formatter` specialization

//{{{ crate includes
#include "prometheus/integrate/detail/display.hpp"
//}}}
//{{{ std includes
#include <compare>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <functional>
#include <initializer_list>
#include <ranges>
#include <span>
#include <utility>
#include <variant>
#include <vector>
//}}}

namespace prometheus::integrate
{

/// Defined in `gauss.hpp`; opaque-declared here because `ConfigIssue` refers to it.
enum class GaussFamily : std::uint8_t;

namespace detail
{
/// Largest polynomial degree representable by `PolynomialDegree`.
///
/// Requested Gaussian rules are currently limited to degree 100, but their actual exactness can
/// be degree 101.
inline constexpr std::size_t MAX_POLYNOMIAL_DEGREE = 101;

/// Largest point count representable by `PointCount`.
inline constexpr std::size_t MAX_POINT_COUNT = 52;
} // namespace detail

//{{{ enum: RuleAxis
/// Coordinate whose Gaussian rule failed adaptive rule-pair validation.
enum class RuleAxis : std::uint8_t
{
    /// The single coordinate of a one-dimensional rule.
    OneDimensional,
    /// The `u` coordinate of a tensor-product rule.
    U,
    /// The `v` coordinate of a tensor-product rule.
    V,
};
//}}}

//{{{ variant: ConfigIssue
/// Alternatives of `ConfigIssue`, one struct per Rust enum variant.
namespace config_issue
{

/// A polynomial degree lies outside the representable range.
struct PolynomialDegreeOutOfRange
{
    std::size_t degree;
    std::size_t maximum;
    bool operator==(const PolynomialDegreeOutOfRange&) const = default;
};

/// A point count lies outside the representable range.
struct PointCountOutOfRange
{
    std::size_t point_count;
    std::size_t minimum;
    std::size_t maximum;
    bool operator==(const PointCountOutOfRange&) const = default;
};

/// One or both interval bounds are non-finite.
struct NonFiniteInterval
{
    double lower;
    double upper;
    bool operator==(const NonFiniteInterval&) const = default;
};

/// Interval bounds are not strictly increasing.
struct InvalidIntervalOrder
{
    double lower;
    double upper;
    bool operator==(const InvalidIntervalOrder&) const = default;
};

/// Absolute and relative tolerances do not form a valid tolerance.
struct InvalidTolerance
{
    double absolute;
    double relative;
    bool operator==(const InvalidTolerance&) const = default;
};

/// An adaptive high-order rule is not more exact than its low-order rule.
struct NonIncreasingRuleExactness
{
    RuleAxis axis;
    std::size_t low;
    std::size_t high;
    bool operator==(const NonIncreasingRuleExactness&) const = default;
};

/// A requested Gaussian rule degree is not supported by the fixed-rule cache.
struct UnsupportedRuleDegree
{
    RuleAxis axis;
    GaussFamily family;
    std::size_t degree;
    std::size_t maximum;
    bool operator==(const UnsupportedRuleDegree&) const = default;
};

/// A subdivision coordinate is non-finite.
struct NonFiniteSubdivision
{
    std::size_t index;
    double value;
    bool operator==(const NonFiniteSubdivision&) const = default;
};

/// A subdivision coordinate is not strictly inside its interval.
struct SubdivisionOutsideInterval
{
    std::size_t index;
    double value;
    double lower;
    double upper;
    bool operator==(const SubdivisionOutsideInterval&) const = default;
};

/// Consecutive subdivision coordinates are not strictly increasing.
struct SubdivisionsNotStrictlyIncreasing
{
    std::size_t previous_index;
    double previous;
    std::size_t index;
    double value;
    bool operator==(const SubdivisionsNotStrictlyIncreasing&) const = default;
};

} // namespace config_issue

/// A single invalid configuration value.
using ConfigIssue = std::variant<config_issue::PolynomialDegreeOutOfRange,
                                 config_issue::PointCountOutOfRange,
                                 config_issue::NonFiniteInterval,
                                 config_issue::InvalidIntervalOrder,
                                 config_issue::InvalidTolerance,
                                 config_issue::NonIncreasingRuleExactness,
                                 config_issue::UnsupportedRuleDegree,
                                 config_issue::NonFiniteSubdivision,
                                 config_issue::SubdivisionOutsideInterval,
                                 config_issue::SubdivisionsNotStrictlyIncreasing>;
//}}}

//{{{ class: ConfigError
/// One or more invalid configuration values.
class ConfigError
{
public:
    explicit ConfigError(std::vector<ConfigIssue> issues) : issues_{std::move(issues)} {}

    static ConfigError one(ConfigIssue issue) { return ConfigError{{std::move(issue)}}; }
    static ConfigError from_issues(std::vector<ConfigIssue> issues)
    {
        return ConfigError{std::move(issues)};
    }

    /// Returns every issue discovered during validation.
    [[nodiscard]] std::span<const ConfigIssue> issues() const noexcept { return issues_; }

    bool operator==(const ConfigError&) const = default;

private:
    std::vector<ConfigIssue> issues_;
};
//}}}

//{{{ class: PolynomialDegree
/// A validated polynomial degree.
class PolynomialDegree
{
public:
    /// Largest representable degree.
    static constexpr std::size_t MAXIMUM = detail::MAX_POLYNOMIAL_DEGREE;

    /// Validates and constructs a polynomial degree; fails when `degree > MAXIMUM`.
    static std::expected<PolynomialDegree, ConfigError> create(std::size_t degree)
    {
        if (degree <= MAXIMUM)
        {
            return PolynomialDegree{degree};
        }
        return std::unexpected{ConfigError::one(
            config_issue::PolynomialDegreeOutOfRange{.degree = degree, .maximum = MAXIMUM})};
    }

    /// Rust `TryFrom<usize>`.
    static std::expected<PolynomialDegree, ConfigError> try_from(std::size_t value)
    {
        return create(value);
    }

    static constexpr PolynomialDegree new_unchecked(std::size_t degree) noexcept
    {
        return PolynomialDegree{degree};
    }

    /// Returns the underlying degree.
    [[nodiscard]] constexpr std::size_t value() const noexcept { return value_; }

    /// Rust `From<PolynomialDegree> for usize`.
    constexpr explicit operator std::size_t() const noexcept { return value_; }

    auto operator<=>(const PolynomialDegree&) const = default;

private:
    constexpr explicit PolynomialDegree(std::size_t value) noexcept : value_{value} {}
    std::size_t value_;
};
//}}}

//{{{ class: PointCount
/// A validated Gaussian point count.
///
/// The generic range covers both supported families. A constructor taking a `GaussFamily`
/// performs the remaining family-specific validation.
class PointCount
{
public:
    /// Smallest representable point count.
    static constexpr std::size_t MINIMUM = 1;
    /// Largest representable point count.
    static constexpr std::size_t MAXIMUM = detail::MAX_POINT_COUNT;

    /// Validates and constructs a point count; fails outside `MINIMUM..=MAXIMUM`.
    static std::expected<PointCount, ConfigError> create(std::size_t point_count)
    {
        if (MINIMUM <= point_count && point_count <= MAXIMUM)
        {
            return PointCount{point_count};
        }
        return std::unexpected{ConfigError::one(config_issue::PointCountOutOfRange{
            .point_count = point_count, .minimum = MINIMUM, .maximum = MAXIMUM})};
    }

    /// Rust `TryFrom<usize>`.
    static std::expected<PointCount, ConfigError> try_from(std::size_t value)
    {
        return create(value);
    }

    static constexpr PointCount new_unchecked(std::size_t point_count) noexcept
    {
        return PointCount{point_count};
    }

    /// Returns the underlying point count.
    [[nodiscard]] constexpr std::size_t value() const noexcept { return value_; }

    constexpr explicit operator std::size_t() const noexcept { return value_; }

    auto operator<=>(const PointCount&) const = default;

private:
    constexpr explicit PointCount(std::size_t value) noexcept : value_{value} {}
    std::size_t value_;
};
//}}}

//{{{ class: Interval
/// A finite, nonempty interval with strictly increasing bounds.
class Interval
{
public:
    /// Validates and constructs an interval; fails unless both bounds are finite and
    /// `lower < upper`.
    static std::expected<Interval, ConfigError> create(double lower, double upper);

    static constexpr Interval new_unchecked(double lower, double upper) noexcept
    {
        return Interval{lower, upper};
    }

    /// Returns the lower bound.
    [[nodiscard]] constexpr double lower() const noexcept { return lower_; }
    /// Returns the upper bound.
    [[nodiscard]] constexpr double upper() const noexcept { return upper_; }
    /// Returns the interval length.
    [[nodiscard]] constexpr double length() const noexcept { return upper_ - lower_; }
    /// Returns the interval bounds as a pair.
    [[nodiscard]] constexpr std::pair<double, double> bounds() const noexcept
    {
        return {lower_, upper_};
    }

    bool operator==(const Interval&) const = default;

private:
    constexpr Interval(double lower, double upper) noexcept : lower_{lower}, upper_{upper} {}
    double lower_;
    double upper_;
};
//}}}

//{{{ class: Rectangle
/// A rectangle composed of validated `u` and `v` intervals.
class Rectangle
{
public:
    /// Constructs a rectangle from validated axis intervals.
    constexpr Rectangle(Interval u, Interval v) noexcept : u_{u}, v_{v} {}

    /// Validates four bounds and constructs a rectangle; the error contains the invalid `u`
    /// and/or `v` interval issues.
    static std::expected<Rectangle, ConfigError>
    from_bounds(double u_lower, double u_upper, double v_lower, double v_upper);

    /// Returns the `u` interval.
    [[nodiscard]] constexpr Interval u() const noexcept { return u_; }
    /// Returns the `v` interval.
    [[nodiscard]] constexpr Interval v() const noexcept { return v_; }

    bool operator==(const Rectangle&) const = default;

private:
    Interval u_;
    Interval v_;
};
//}}}

//{{{ class: Tolerance
/// Absolute and relative error tolerances.
class Tolerance
{
public:
    /// Validates and constructs an absolute-plus-relative tolerance; fails unless both
    /// components are finite and nonnegative and at least one component is positive.
    static std::expected<Tolerance, ConfigError> create(double absolute, double relative);

    /// Constructs a purely absolute tolerance.
    static std::expected<Tolerance, ConfigError> absolute(double absolute)
    {
        return create(absolute, 0.0);
    }

    /// Constructs a purely relative tolerance.
    static std::expected<Tolerance, ConfigError> relative(double relative)
    {
        return create(0.0, relative);
    }

    /// Returns the absolute component.
    [[nodiscard]] constexpr double absolute_value() const noexcept { return absolute_; }
    /// Returns the relative component.
    [[nodiscard]] constexpr double relative_value() const noexcept { return relative_; }

    bool operator==(const Tolerance&) const = default;

private:
    constexpr Tolerance(double absolute, double relative) noexcept
        : absolute_{absolute}, relative_{relative}
    {
    }
    double absolute_;
    double relative_;
};
//}}}

//{{{ class: RefinementDepth
/// Maximum adaptive-refinement depth.
///
/// A depth of zero evaluates the initial regions without permitting refinement.
class RefinementDepth
{
public:
    /// Default maximum refinement depth used by adaptive builders.
    static constexpr std::size_t DEFAULT = 32;

    /// Rust `Default`.
    constexpr RefinementDepth() noexcept : value_{DEFAULT} {}
    /// Constructs a refinement depth. Zero is valid. (Rust `new` / `From<usize>`.)
    constexpr explicit RefinementDepth(std::size_t depth) noexcept : value_{depth} {}

    /// Returns the underlying depth.
    [[nodiscard]] constexpr std::size_t value() const noexcept { return value_; }

    constexpr explicit operator std::size_t() const noexcept { return value_; }

    auto operator<=>(const RefinementDepth&) const = default;

private:
    std::size_t value_;
};
//}}}

//{{{ class: AxisDepths
/// Maximum adaptive-refinement depths for two axes.
class AxisDepths
{
public:
    /// Rust `Default`: `uniform(RefinementDepth{})`.
    constexpr AxisDepths() noexcept = default;

    /// Constructs independent `u` and `v` refinement depths.
    constexpr AxisDepths(RefinementDepth u, RefinementDepth v) noexcept : u_{u}, v_{v} {}

    /// Constructs independent axis depths from their underlying integer values.
    static constexpr AxisDepths from_values(std::size_t u, std::size_t v) noexcept
    {
        return {RefinementDepth{u}, RefinementDepth{v}};
    }

    /// Constructs equal depths for both axes.
    static constexpr AxisDepths uniform(RefinementDepth depth) noexcept { return {depth, depth}; }

    /// Returns the `u`-axis depth.
    [[nodiscard]] constexpr RefinementDepth u() const noexcept { return u_; }
    /// Returns the `v`-axis depth.
    [[nodiscard]] constexpr RefinementDepth v() const noexcept { return v_; }

    bool operator==(const AxisDepths&) const = default;

private:
    RefinementDepth u_{};
    RefinementDepth v_{};
};
//}}}

namespace detail
{

//{{{ fun: validate_subdivisions
/// Validates collected subdivision coordinates, reporting *every* issue found.
std::expected<std::vector<double>, ConfigError> validate_subdivisions(Interval interval,
                                                                      std::vector<double> points);
//}}}

//{{{ concept: SubdivisionRange
/// Rust `I: IntoIterator<Item = f64>`.
template <class R>
concept SubdivisionRange =
    std::ranges::input_range<R> && std::convertible_to<std::ranges::range_reference_t<R>, double>;

template <SubdivisionRange R>
std::vector<double> collect_points(R&& points)
{
    std::vector<double> out;
    if constexpr (std::ranges::sized_range<R>)
    {
        out.reserve(static_cast<std::size_t>(std::ranges::size(points)));
    }
    for (auto&& point : points)
    {
        out.push_back(static_cast<double>(point));
    }
    return out;
}
//}}}

//{{{ fun: merge_axis_subdivisions
/// Shared body of the two-dimensional `subdivisions` builders: on success returns the
/// validated `(u, v)` lists; when both axes fail, the issues are concatenated `u` first.
std::expected<std::pair<std::vector<double>, std::vector<double>>, ConfigError>
validate_axis_subdivisions(Rectangle domain, std::vector<double> u, std::vector<double> v);
//}}}

} // namespace detail
} // namespace prometheus::integrate

//{{{ collection: std::formatter specializations (Rust `Display`)
PROMETHEUS_INTEGRATE_DECLARE_FORMATTER(prometheus::integrate::RuleAxis);
PROMETHEUS_INTEGRATE_DECLARE_FORMATTER(prometheus::integrate::ConfigIssue);
PROMETHEUS_INTEGRATE_DECLARE_FORMATTER(prometheus::integrate::ConfigError);
PROMETHEUS_INTEGRATE_DECLARE_FORMATTER(prometheus::integrate::PolynomialDegree);
PROMETHEUS_INTEGRATE_DECLARE_FORMATTER(prometheus::integrate::PointCount);
PROMETHEUS_INTEGRATE_DECLARE_FORMATTER(prometheus::integrate::Interval);
PROMETHEUS_INTEGRATE_DECLARE_FORMATTER(prometheus::integrate::Rectangle);
PROMETHEUS_INTEGRATE_DECLARE_FORMATTER(prometheus::integrate::Tolerance);
PROMETHEUS_INTEGRATE_DECLARE_FORMATTER(prometheus::integrate::RefinementDepth);
PROMETHEUS_INTEGRATE_DECLARE_FORMATTER(prometheus::integrate::AxisDepths);
//}}}

//{{{ collection: std::hash specializations (Rust `Hash`)
template <>
struct std::hash<prometheus::integrate::PolynomialDegree>
{
    std::size_t operator()(prometheus::integrate::PolynomialDegree d) const noexcept
    {
        return std::hash<std::size_t>{}(d.value());
    }
};

template <>
struct std::hash<prometheus::integrate::PointCount>
{
    std::size_t operator()(prometheus::integrate::PointCount c) const noexcept
    {
        return std::hash<std::size_t>{}(c.value());
    }
};

template <>
struct std::hash<prometheus::integrate::RefinementDepth>
{
    std::size_t operator()(prometheus::integrate::RefinementDepth d) const noexcept
    {
        return std::hash<std::size_t>{}(d.value());
    }
};

template <>
struct std::hash<prometheus::integrate::AxisDepths>
{
    std::size_t operator()(prometheus::integrate::AxisDepths d) const noexcept
    {
        const std::size_t u = d.u().value();
        const std::size_t v = d.v().value();
        return std::hash<std::size_t>{}(u) ^ (std::hash<std::size_t>{}(v) << 1U);
    }
};
//}}}
