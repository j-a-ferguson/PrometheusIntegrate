#pragma once
//! Numerical quadrature rules for one- and two-dimensional real-valued functions.
//!
//! The library provides fixed Gauss quadrature rules and adaptive algorithms built from pairs of
//! fixed rules. The dimensionality is encoded in the names of the public types and functions.
//!
//! # Fixed quadrature
//!
//! ```cpp
//! using namespace prometheus::integrate;
//! auto domain = Interval::create(-1.0, 1.0).value();
//! auto degree = PolynomialDegree::create(5).value();
//! auto rule = GaussRule::for_degree(GaussFamily::Legendre, degree).value();
//! auto quadrature = FixedQuadrature1d::builder(domain, rule).build();
//! double integral = quadrature.integrate([](double x) { return x * x; }).value();
//! ```
//!
//! # Adaptive quadrature
//!
//! ```cpp
//! auto low  = GaussRule::for_degree(GaussFamily::Legendre, PolynomialDegree::create(3).value());
//! auto high = GaussRule::for_degree(GaussFamily::Legendre, PolynomialDegree::create(9).value());
//! auto quadrature = AdaptiveQuadrature1d::builder(Interval::create(0.0, 1.0).value(),
//!                                                 low.value(), high.value(),
//!                                                 Tolerance::create(1e-10, 1e-10).value())
//!                       .build().value();
//! auto result = quadrature.integrate([](double x) { return std::exp(x); }).value();
//! ```

//{{{ crate includes
#include "prometheus/integrate/adaptive/d1.hpp"
#include "prometheus/integrate/adaptive/d2.hpp"
#include "prometheus/integrate/common.hpp"
#include "prometheus/integrate/config.hpp"
#include "prometheus/integrate/fixed/d1.hpp"
#include "prometheus/integrate/fixed/d2.hpp"
#include "prometheus/integrate/gauss.hpp"
#include "prometheus/integrate/integration.hpp"
//}}}
//{{{ std includes
#include <expected>
#include <utility>
//}}}

namespace prometheus::integrate
{

//{{{ collection: public exports (Rust `pub use ... as ...`)
using AdaptiveQuadrature1d = adaptive::d1::AdaptiveQuadrature;
using AdaptiveQuadratureBuilder1d = adaptive::d1::Builder;
using AdaptiveQuadrature2d = adaptive::d2::AdaptiveQuadrature;
using AdaptiveQuadratureBuilder2d = adaptive::d2::Builder;

using FixedQuadratureBuilder1d = fixed::d1::Builder;
using FixedNode1d = fixed::d1::Node;
using FixedQuadrature1d = fixed::d1::Quadrature;
using FixedQuadratureBuilder2d = fixed::d2::Builder;
using FixedNode2d = fixed::d2::Node;
using FixedQuadrature2d = fixed::d2::Quadrature;
using TensorRule2d = fixed::d2::TensorRule;

/// Function templates cannot be aliased, so the renamed free functions are thin forwarders.
template <Integrand1d F>
std::expected<double, IntegrationError> fixed_quad_1d(F&& f, FixedQuadratureBuilder1d builder)
{
    return fixed::d1::fixed_quad(std::forward<F>(f), std::move(builder));
}

template <Integrand2d F>
std::expected<double, IntegrationError> fixed_quad_2d(F&& f, FixedQuadratureBuilder2d builder)
{
    return fixed::d2::fixed_quad(std::forward<F>(f), std::move(builder));
}

template <Integrand1d F>
std::expected<AdaptiveResult, OptionsError> adaptive_quad_1d(F&& f,
                                                             AdaptiveQuadratureBuilder1d builder)
{
    return adaptive::d1::adaptive_quad(std::forward<F>(f), std::move(builder));
}

template <Integrand2d F>
std::expected<AdaptiveResult, OptionsError> adaptive_quad_2d(F&& f,
                                                             AdaptiveQuadratureBuilder2d builder)
{
    return adaptive::d2::adaptive_quad(std::forward<F>(f), std::move(builder));
}
//}}}

} // namespace prometheus::integrate
