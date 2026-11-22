//! `Display` for integration results, integration errors and the one-shot `OptionsError`.

#include "prometheus/integrate/integration.hpp"
#include "prometheus/integrate/common.hpp"

#include <format>
#include <string>
#include <variant>

namespace ti = prometheus::integrate;
namespace ep = prometheus::integrate::evaluation_point;
namespace ie = prometheus::integrate::integration_error;
using ti::detail::overloaded;
using ti::detail::RustFloat;

std::string ti::detail::display(const ti::EvaluationPoint& value)
{
    return std::visit(
        overloaded{
            [&](const ep::OneDimensional& p) { return std::format("x = {}", RustFloat{p.x}); },
            [&](const ep::TwoDimensional& p)
            { return std::format("(u, v) = ({}, {})", RustFloat{p.u}, RustFloat{p.v}); },
        },
        value);
}

std::string ti::detail::display(const ti::IntegrationError& value)
{
    return std::visit(
        overloaded{
            [&](const ie::NonFiniteIntegrand& e)
            {
                return std::format(
                    "integrand returned non-finite value {} at {}", RustFloat{e.value}, e.point);
            },
            [&](const ie::MaxDepthReached&)
            { return std::format("maximum refinement depth reached before convergence"); },
            [&](const ie::NonProgressingInterval& e)
            {
                return std::format(
                    "cannot subdivide interval {} because its midpoint is not interior",
                    e.interval);
            },
            [&](const ie::NonProgressingRectangle& e)
            {
                return std::format("cannot subdivide rectangle {} because an active-axis "
                                   "midpoint is not interior",
                                   e.rectangle);
            },
        },
        value);
}

std::string ti::detail::display(const ti::OptionsError& value)
{
    // `#[error(transparent)]` on every variant: forward to the wrapped error's Display.
    return std::visit([](const auto& inner) { return std::format("{}", inner); }, value);
}
