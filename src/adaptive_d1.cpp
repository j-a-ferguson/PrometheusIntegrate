//! Adaptive quadrature for one-dimensional real-valued functions.

#include "prometheus/integrate/adaptive/d1.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace prometheus::integrate::adaptive::d1
{

//{{{ impl: Builder
std::expected<Builder, ConfigError> Builder::set_subdivisions(std::vector<double> points) &&
{
    auto validated = integrate::detail::validate_subdivisions(domain_, std::move(points));
    if (!validated)
    {
        return std::unexpected{std::move(validated).error()};
    }
    subdivisions_ = std::move(*validated);
    return std::move(*this);
}

std::expected<AdaptiveQuadrature, ConfigError> Builder::build() &&
{
    const std::size_t low_exactness = low_rule_.exactness().value();
    const std::size_t high_exactness = high_rule_.exactness().value();
    if (high_exactness <= low_exactness)
    {
        return std::unexpected{ConfigError::from_issues({config_issue::NonIncreasingRuleExactness{
            .axis = RuleAxis::OneDimensional, .low = low_exactness, .high = high_exactness}})};
    }

    auto low = detail::FixedQuadrature::builder(domain_, std::move(low_rule_)).build();
    auto high = detail::FixedQuadrature::builder(domain_, std::move(high_rule_)).build();
    return AdaptiveQuadrature{
        domain_, std::move(low), std::move(high), tolerance_, max_depth_, std::move(subdivisions_)};
}
//}}}

namespace detail
{

//{{{ fun: converged
bool converged(const AdaptiveResult& result, Tolerance tolerance) noexcept
{
    return result.error_estimate() <=
           tolerance.absolute_value() + tolerance.relative_value() * std::abs(result.integral());
}
//}}}

//{{{ fun: midpoint
double midpoint(Interval interval) noexcept
{
    const double direct = interval.lower() + 0.5 * (interval.upper() - interval.lower());
    if (std::isfinite(direct))
    {
        return direct;
    }
    return 0.5 * interval.lower() + 0.5 * interval.upper();
}
//}}}

} // namespace detail
} // namespace prometheus::integrate::adaptive::d1
