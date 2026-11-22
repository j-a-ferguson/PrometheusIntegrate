//! Adaptive tensor-product quadrature for two-dimensional real-valued functions.

#include "prometheus/integrate/adaptive/d2.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace prometheus::integrate::adaptive::d2
{

//{{{ impl: Builder
std::expected<Builder, ConfigError> Builder::set_subdivisions(std::vector<double> u,
                                                              std::vector<double> v) &&
{
    auto validated =
        integrate::detail::validate_axis_subdivisions(domain_, std::move(u), std::move(v));
    if (!validated)
    {
        return std::unexpected{std::move(validated).error()};
    }
    u_subdivisions_ = std::move(validated->first);
    v_subdivisions_ = std::move(validated->second);
    return std::move(*this);
}

std::expected<AdaptiveQuadrature, ConfigError> Builder::build() &&
{
    std::vector<ConfigIssue> issues;
    detail::validate_rule_pair(low_rule_.u(), high_rule_.u(), RuleAxis::U, issues);
    detail::validate_rule_pair(low_rule_.v(), high_rule_.v(), RuleAxis::V, issues);
    if (!issues.empty())
    {
        return std::unexpected{ConfigError::from_issues(std::move(issues))};
    }

    auto low = detail::FixedQuadrature::builder(domain_, std::move(low_rule_)).build();
    auto high = detail::FixedQuadrature::builder(domain_, std::move(high_rule_)).build();
    return AdaptiveQuadrature{domain_,
                              std::move(low),
                              std::move(high),
                              tolerance_,
                              max_depth_,
                              std::move(u_subdivisions_),
                              std::move(v_subdivisions_)};
}
//}}}

namespace detail
{

//{{{ fun: validate_rule_pair
void validate_rule_pair(const GaussRule& low,
                        const GaussRule& high,
                        RuleAxis axis,
                        std::vector<ConfigIssue>& issues)
{
    const std::size_t low_exactness = low.exactness().value();
    const std::size_t high_exactness = high.exactness().value();
    if (high_exactness <= low_exactness)
    {
        issues.emplace_back(config_issue::NonIncreasingRuleExactness{
            .axis = axis, .low = low_exactness, .high = high_exactness});
    }
}
//}}}

//{{{ fun: axis_bounds
std::vector<double> axis_bounds(Interval interval, std::span<const double> subdivisions)
{
    std::vector<double> bounds;
    bounds.reserve(subdivisions.size() + 2);
    bounds.push_back(interval.lower());
    bounds.insert(bounds.end(), subdivisions.begin(), subdivisions.end());
    bounds.push_back(interval.upper());
    return bounds;
}
//}}}

namespace
{
//{{{ fun: split_axis
std::vector<Interval> split_axis(Interval interval, std::optional<double> midpoint)
{
    if (midpoint)
    {
        return {Interval::new_unchecked(interval.lower(), *midpoint),
                Interval::new_unchecked(*midpoint, interval.upper())};
    }
    return {interval};
}
//}}}
} // namespace

//{{{ fun: child_domains
std::vector<Rectangle>
child_domains(Rectangle domain, std::optional<double> u_midpoint, std::optional<double> v_midpoint)
{
    const auto u_intervals = split_axis(domain.u(), u_midpoint);
    const auto v_intervals = split_axis(domain.v(), v_midpoint);
    std::vector<Rectangle> children;
    children.reserve(u_intervals.size() * v_intervals.size());
    for (const Interval& u : u_intervals)
    {
        for (const Interval& v : v_intervals)
        {
            children.emplace_back(u, v);
        }
    }
    return children;
}
//}}}

} // namespace detail
} // namespace prometheus::integrate::adaptive::d2
