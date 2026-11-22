//! Fixed quadrature for one-dimensional real-valued functions.

#include "prometheus/integrate/fixed/d1.hpp"

#include <utility>
#include <vector>

namespace prometheus::integrate::fixed::d1
{

//{{{ fun: mapped_nodes
std::vector<Node>
detail::mapped_nodes(Interval domain, const GaussRule& rule, std::span<const double> subdivisions)
{
    const std::size_t point_count = rule.point_count().value();
    std::vector<Node> nodes;
    nodes.reserve(point_count * (subdivisions.size() + 1));
    const auto [reference_lower, reference_upper] = range(rule.family());
    const double reference_length = reference_upper - reference_lower;
    const auto points = rule.points();
    const auto weights = rule.weights();

    double lower = domain.lower();
    // Rust: `subdivisions.iter().chain(once(domain.upper()))`.
    const auto map_segment = [&](double upper)
    {
        const double scale = (upper - lower) / reference_length;
        for (std::size_t k = 0; k < point_count; ++k)
        {
            nodes.push_back(
                Node{lower + scale * (points[k] - reference_lower), scale * weights[k]});
        }
        lower = upper;
    };
    for (const double upper : subdivisions)
    {
        map_segment(upper);
    }
    map_segment(domain.upper());

    return nodes;
}
//}}}

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

Quadrature Builder::build() &&
{
    auto nodes = detail::mapped_nodes(domain_, rule_, subdivisions_);
    return Quadrature{domain_, std::move(rule_), std::move(subdivisions_), std::move(nodes)};
}
//}}}

//{{{ impl: From<GaussRule> for Quadrature
Quadrature::Quadrature(GaussRule rule)
    : Quadrature{[&]
                 {
                     const auto [lower, upper] = range(rule.family());
                     return builder(Interval::new_unchecked(lower, upper), std::move(rule)).build();
                 }()}
{
}
//}}}

} // namespace prometheus::integrate::fixed::d1
