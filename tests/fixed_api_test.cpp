//! Port of `tests/fixed_api.rs`.

#include "test_support.hpp"

#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <ranges>
#include <type_traits>
#include <variant>
#include <vector>

using namespace prometheus::integrate;
using test_support::count;
using test_support::degree;
using test_support::interval;
using test_support::unwrap;
using test_support::unwrap_err;

TEST(FixedApi, TypedNodesHaveTheSameCompactLayoutAsPackedF64Values)
{
    // Rust `#[repr(C)]` + `size_of` / `align_of`.
    static_assert(std::is_standard_layout_v<FixedNode1d>);
    static_assert(std::is_standard_layout_v<FixedNode2d>);
    EXPECT_EQ(sizeof(FixedNode1d), 2 * sizeof(double));
    EXPECT_EQ(alignof(FixedNode1d), alignof(double));
    EXPECT_EQ(sizeof(FixedNode2d), 3 * sizeof(double));
    EXPECT_EQ(alignof(FixedNode2d), alignof(double));
}

TEST(FixedApi, OneDimensionalBuilderMapsAndIteratesTypedNodes)
{
    const auto domain = interval(2.0, 6.0);
    auto rule = unwrap(GaussRule::for_degree(GaussFamily::Legendre, degree(1)));
    const auto quadrature =
        unwrap(FixedQuadrature1d::builder(domain, std::move(rule)).subdivisions({3.0, 4.5}))
            .build();

    EXPECT_EQ(quadrature.domain(), domain);
    EXPECT_TRUE(std::ranges::equal(quadrature.subdivision_points(), std::vector{3.0, 4.5}));
    EXPECT_EQ(quadrature.point_count(), 3U);
    EXPECT_EQ(static_cast<std::size_t>(std::ranges::distance(quadrature)),
              quadrature.nodes().size());

    std::vector<double> points;
    std::vector<double> weights;
    for (const FixedNode1d& node : quadrature)
    {
        points.push_back(node.point());
        weights.push_back(node.weight());
    }
    EXPECT_EQ(points, (std::vector{2.5, 3.75, 5.25}));
    EXPECT_EQ(weights, (std::vector{1.0, 1.5, 1.5}));
}

TEST(FixedApi, OneDimensionalIntegrationAcceptsFnMutAndReportsNonfiniteValues)
{
    auto rule = unwrap(GaussRule::for_degree(GaussFamily::Legendre, degree(9)));
    const auto quadrature =
        FixedQuadrature1d::builder(interval(-1.0, 1.0), std::move(rule)).build();
    std::size_t evaluations = 0;

    const double integral = unwrap(quadrature.integrate(
        [&](double x)
        {
            ++evaluations;
            return test_support::powi(x, 4);
        }));
    EXPECT_EQ(evaluations, quadrature.point_count());
    EXPECT_NEAR(integral, 2.0 / 5.0, 1e-14);

    const double failing_point = quadrature.nodes()[2].point();
    const auto error = unwrap_err(quadrature.integrate(
        [&](double x)
        { return x == failing_point ? std::numeric_limits<double>::quiet_NaN() : x; }));
    const auto* e = std::get_if<integration_error::NonFiniteIntegrand>(&error);
    ASSERT_NE(e, nullptr);
    const auto* point = std::get_if<evaluation_point::OneDimensional>(&e->point);
    ASSERT_NE(point, nullptr);
    EXPECT_EQ(point->x, failing_point);
    EXPECT_TRUE(std::isnan(e->value));
}

TEST(FixedApi, IntegrateOverRemapsSubdivisionsProportionally)
{
    auto rule = unwrap(GaussRule::with_point_count(GaussFamily::Lobatto, count(2)));
    const auto quadrature = unwrap(FixedQuadrature1d::builder(interval(0.0, 10.0), std::move(rule))
                                       .subdivisions({2.0, 7.0}))
                                .build();
    std::vector<double> evaluated_points;

    const double integral = unwrap(quadrature.integrate_over(interval(100.0, 200.0),
                                                             [&](double x)
                                                             {
                                                                 evaluated_points.push_back(x);
                                                                 return 1.0;
                                                             }));

    EXPECT_NEAR(integral, 100.0, 1e-13);
    EXPECT_EQ(evaluated_points, (std::vector{100.0, 120.0, 120.0, 170.0, 170.0, 200.0}));
}

TEST(FixedApi, TwoDimensionalBuilderMapsTypedTensorNodes)
{
    const Rectangle rectangle{interval(0.0, 2.0), interval(-1.0, 1.0)};
    TensorRule2d tensor_rule{unwrap(GaussRule::for_degree(GaussFamily::Legendre, degree(3))),
                             unwrap(GaussRule::for_degree(GaussFamily::Legendre, degree(3)))};
    const auto quadrature = unwrap(FixedQuadrature2d::builder(rectangle, std::move(tensor_rule))
                                       .subdivisions(std::vector{1.0}, std::views::empty<double>))
                                .build();
    std::size_t evaluations = 0;

    const double integral = unwrap(quadrature.integrate(
        [&](double u, double v)
        {
            ++evaluations;
            return u * u + v * v;
        }));

    EXPECT_EQ(quadrature.domain(), rectangle);
    EXPECT_TRUE(std::ranges::equal(quadrature.u_subdivision_points(), std::vector{1.0}));
    EXPECT_TRUE(quadrature.v_subdivision_points().empty());
    EXPECT_EQ(evaluations, quadrature.point_count());
    EXPECT_EQ(static_cast<std::size_t>(std::ranges::distance(quadrature)),
              quadrature.nodes().size());
    EXPECT_NEAR(integral, 20.0 / 3.0, 1e-13);
}

TEST(FixedApi, TwoDimensionalIntegrationReportsTheEvaluationCoordinates)
{
    const Rectangle rectangle{interval(-1.0, 1.0), interval(-1.0, 1.0)};
    TensorRule2d tensor_rule{unwrap(GaussRule::for_degree(GaussFamily::Legendre, degree(1))),
                             unwrap(GaussRule::for_degree(GaussFamily::Legendre, degree(1)))};
    const auto quadrature = FixedQuadrature2d::builder(rectangle, std::move(tensor_rule)).build();

    const auto error = unwrap_err(quadrature.integrate(
        [](double, double) { return std::numeric_limits<double>::infinity(); }));
    const auto* e = std::get_if<integration_error::NonFiniteIntegrand>(&error);
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(e->point, (EvaluationPoint{evaluation_point::TwoDimensional{.u = 0.0, .v = 0.0}}));
    EXPECT_TRUE(std::isinf(e->value));
}
