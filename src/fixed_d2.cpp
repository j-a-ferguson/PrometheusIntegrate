//! Fixed tensor-product quadrature for two-dimensional real-valued functions.

#include "prometheus/integrate/fixed/d2.hpp"

#include <utility>
#include <vector>

namespace prometheus::integrate::fixed::d2
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

Quadrature Builder::build() &&
{
    const auto u_nodes = d1::detail::mapped_nodes(domain_.u(), rule_.u(), u_subdivisions_);
    const auto v_nodes = d1::detail::mapped_nodes(domain_.v(), rule_.v(), v_subdivisions_);
    std::vector<Node> nodes;
    nodes.reserve(u_nodes.size() * v_nodes.size());

    for (const auto& u : u_nodes)
    {
        for (const auto& v : v_nodes)
        {
            nodes.push_back(Node{u.point(), v.point(), u.weight() * v.weight()});
        }
    }

    return Quadrature{domain_,
                      std::move(rule_),
                      std::move(u_subdivisions_),
                      std::move(v_subdivisions_),
                      std::move(nodes)};
}
//}}}

} // namespace prometheus::integrate::fixed::d2
