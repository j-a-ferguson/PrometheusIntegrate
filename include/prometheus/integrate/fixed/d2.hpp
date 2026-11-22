#pragma once
//! Fixed tensor-product quadrature for two-dimensional real-valued functions.

//{{{ crate includes
#include "prometheus/integrate/config.hpp"
#include "prometheus/integrate/fixed/d1.hpp"
#include "prometheus/integrate/gauss.hpp"
#include "prometheus/integrate/integration.hpp"
//}}}
//{{{ std includes
#include <cmath>
#include <cstddef>
#include <expected>
#include <initializer_list>
#include <span>
#include <utility>
#include <vector>
//}}}

namespace prometheus::integrate::fixed::d2
{

class Builder;
class Quadrature;

//{{{ class: TensorRule
/// A tensor product of independent Gaussian rules on the `u` and `v` axes.
class TensorRule
{
public:
    TensorRule(GaussRule u, GaussRule v) : u_{std::move(u)}, v_{std::move(v)} {}

    /// Returns the `u`-axis Gaussian rule.
    [[nodiscard]] const GaussRule& u() const noexcept { return u_; }
    /// Returns the `v`-axis Gaussian rule.
    [[nodiscard]] const GaussRule& v() const noexcept { return v_; }

    bool operator==(const TensorRule&) const = default;

private:
    GaussRule u_;
    GaussRule v_;
};
//}}}

//{{{ class: Node
/// A mapped point and weight for two-dimensional fixed quadrature (`#[repr(C)]`: 3 doubles).
class Node
{
public:
    /// Returns the `u` coordinate.
    [[nodiscard]] constexpr double u() const noexcept { return u_; }
    /// Returns the `v` coordinate.
    [[nodiscard]] constexpr double v() const noexcept { return v_; }
    /// Returns the tensor-product weight.
    [[nodiscard]] constexpr double weight() const noexcept { return weight_; }

    bool operator==(const Node&) const = default;

private:
    friend class Builder;
    constexpr Node(double u, double v, double weight) noexcept : u_{u}, v_{v}, weight_{weight} {}

    double u_;
    double v_;
    double weight_;
};
//}}}

//{{{ class: Builder
/// Consuming builder for a two-dimensional fixed quadrature.
class Builder
{
public:
    /// Replaces the interior subdivision coordinates on both axes. Empty input on either axis
    /// means no subdivisions on that axis. Fails with every detected issue, `u` first.
    template <integrate::detail::SubdivisionRange U, integrate::detail::SubdivisionRange V>
    std::expected<Builder, ConfigError> subdivisions(U&& u_points, V&& v_points) &&
    {
        return std::move(*this).set_subdivisions(
            integrate::detail::collect_points(std::forward<U>(u_points)),
            integrate::detail::collect_points(std::forward<V>(v_points)));
    }

    std::expected<Builder, ConfigError> subdivisions(std::initializer_list<double> u_points,
                                                     std::initializer_list<double> v_points) &&
    {
        return std::move(*this).set_subdivisions(std::vector<double>(u_points),
                                                 std::vector<double>(v_points));
    }

    /// Maps the tensor rule onto the configured rectangle and subdivisions.
    [[nodiscard]] Quadrature build() &&;

    bool operator==(const Builder&) const = default;

private:
    friend class Quadrature;
    Builder(Rectangle domain, TensorRule rule) : domain_{domain}, rule_{std::move(rule)} {}

    std::expected<Builder, ConfigError> set_subdivisions(std::vector<double> u,
                                                         std::vector<double> v) &&;

    Rectangle domain_;
    TensorRule rule_;
    std::vector<double> u_subdivisions_;
    std::vector<double> v_subdivisions_;
};
//}}}

//{{{ class: Quadrature
/// A reusable two-dimensional fixed tensor-product quadrature.
class Quadrature
{
public:
    using const_iterator = std::vector<Node>::const_iterator;

    /// Starts a consuming builder for `domain` and `rule`.
    [[nodiscard]] static Builder builder(Rectangle domain, TensorRule rule)
    {
        return Builder{domain, std::move(rule)};
    }

    /// Returns the configured rectangular domain.
    [[nodiscard]] Rectangle domain() const noexcept { return domain_; }
    /// Returns the tensor-product reference rule.
    [[nodiscard]] const TensorRule& rule() const noexcept { return rule_; }
    /// Returns the validated `u`-axis subdivision coordinates.
    [[nodiscard]] std::span<const double> u_subdivision_points() const noexcept
    {
        return u_subdivisions_;
    }
    /// Returns the validated `v`-axis subdivision coordinates.
    [[nodiscard]] std::span<const double> v_subdivision_points() const noexcept
    {
        return v_subdivisions_;
    }
    /// Returns all mapped tensor-product nodes.
    [[nodiscard]] std::span<const Node> nodes() const noexcept { return nodes_; }
    /// Returns the total number of mapped tensor-product points.
    [[nodiscard]] std::size_t point_count() const noexcept { return nodes_.size(); }

    [[nodiscard]] const_iterator begin() const noexcept { return nodes_.begin(); }
    [[nodiscard]] const_iterator end() const noexcept { return nodes_.end(); }

    /// Integrates over the configured rectangle.
    template <Integrand2d F>
    std::expected<double, IntegrationError> integrate(F&& f) const
    {
        double integral = 0.0;
        for (const Node& node : nodes_)
        {
            const double value = static_cast<double>(f(node.u(), node.v()));
            if (!std::isfinite(value))
            {
                return std::unexpected{integration_error::NonFiniteIntegrand{
                    .point = evaluation_point::TwoDimensional{node.u(), node.v()}, .value = value}};
            }
            integral += node.weight() * value;
        }
        return integral;
    }

    /// Integrates over another rectangle by proportionally remapping every node and subdivision.
    template <Integrand2d F>
    std::expected<double, IntegrationError> integrate_over(Rectangle domain, F&& f) const
    {
        const double u_scale = domain.u().length() / domain_.u().length();
        const double v_scale = domain.v().length() / domain_.v().length();
        double integral = 0.0;

        for (const Node& node : nodes_)
        {
            const double u = domain.u().lower() + u_scale * (node.u() - domain_.u().lower());
            const double v = domain.v().lower() + v_scale * (node.v() - domain_.v().lower());
            const double value = static_cast<double>(f(u, v));
            if (!std::isfinite(value))
            {
                return std::unexpected{integration_error::NonFiniteIntegrand{
                    .point = evaluation_point::TwoDimensional{u, v}, .value = value}};
            }
            integral += u_scale * v_scale * node.weight() * value;
        }
        return integral;
    }

    bool operator==(const Quadrature&) const = default;

private:
    friend class Builder;
    Quadrature(Rectangle domain,
               TensorRule rule,
               std::vector<double> u_subdivisions,
               std::vector<double> v_subdivisions,
               std::vector<Node> nodes)
        : domain_{domain}, rule_{std::move(rule)}, u_subdivisions_{std::move(u_subdivisions)},
          v_subdivisions_{std::move(v_subdivisions)}, nodes_{std::move(nodes)}
    {
    }

    Rectangle domain_;
    TensorRule rule_;
    std::vector<double> u_subdivisions_;
    std::vector<double> v_subdivisions_;
    std::vector<Node> nodes_;
};
//}}}

//{{{ fun: fixed_quad
/// Builds a two-dimensional fixed quadrature and integrates `f` once.
template <Integrand2d F>
std::expected<double, IntegrationError> fixed_quad(F&& f, Builder builder)
{
    return std::move(builder).build().integrate(std::forward<F>(f));
}
//}}}

} // namespace prometheus::integrate::fixed::d2
