//! Port of `tests/fixed.rs`.

#include "test_support.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

using namespace prometheus::integrate;
using test_support::degree;
using test_support::unwrap;

namespace
{

constexpr double MAX_RELATIVE_ERROR = 1e-14;

//{{{ collection: test data
struct Polynomial1d
{
    std::string name;
    std::vector<double> coeffs;
    double integral;
};

struct Data1d
{
    std::array<double, 2> range;
    std::vector<Polynomial1d> polynomials;
};

struct Polynomial2d
{
    std::string name;
    std::vector<double> coeffs_u;
    std::vector<double> coeffs_v;
    double integral;
};

struct Data2d
{
    std::array<double, 4> range;
    std::vector<Polynomial2d> polynomials;
};

/// `#[serde(flatten)] BTreeMap<String, Polynomial>`: every key except `range` is a polynomial.
Data1d data_1d()
{
    const auto json = test_support::load_asset("poly-integrals-1d.json").at("values");
    Data1d data{json.at("range").get<std::array<double, 2>>(), {}};
    for (const auto& [key, value] : json.items())
    {
        if (key != "range")
        {
            data.polynomials.push_back({key,
                                        value.at("coeffs").get<std::vector<double>>(),
                                        value.at("integral").get<double>()});
        }
    }
    return data;
}

Data2d data_2d()
{
    const auto json = test_support::load_asset("poly-integrals-2d.json").at("values");
    Data2d data{json.at("range").get<std::array<double, 4>>(), {}};
    for (const auto& [key, value] : json.items())
    {
        if (key != "range")
        {
            data.polynomials.push_back({key,
                                        value.at("coeffs_u").get<std::vector<double>>(),
                                        value.at("coeffs_v").get<std::vector<double>>(),
                                        value.at("integral").get<double>()});
        }
    }
    return data;
}
//}}}

//{{{ collection: helpers
double evaluate_polynomial(const std::vector<double>& coeffs, double x)
{
    double sum = 0.0;
    for (std::size_t power = 0; power < coeffs.size(); ++power)
    {
        sum += coeffs[power] * test_support::powi(x, static_cast<int>(power));
    }
    return sum;
}

FixedQuadratureBuilder1d
builder_1d(GaussFamily family, std::size_t exactness, std::array<double, 2> bounds)
{
    auto rule = unwrap(GaussRule::for_degree(family, degree(exactness)));
    return FixedQuadrature1d::builder(test_support::interval(bounds[0], bounds[1]),
                                      std::move(rule));
}

FixedQuadratureBuilder2d builder_2d(GaussFamily family,
                                    std::size_t u_exactness,
                                    std::size_t v_exactness,
                                    std::array<double, 4> bounds)
{
    auto u_rule = unwrap(GaussRule::for_degree(family, degree(u_exactness)));
    auto v_rule = unwrap(GaussRule::for_degree(family, degree(v_exactness)));
    const auto domain = unwrap(Rectangle::from_bounds(bounds[0], bounds[1], bounds[2], bounds[3]));
    return FixedQuadrature2d::builder(domain, TensorRule2d{std::move(u_rule), std::move(v_rule)});
}
//}}}

} // namespace

//{{{ collection: one-dimensional tests
TEST(Fixed, OnePointLegendreRuleIntegratesLinearFunction)
{
    const double integral = unwrap(fixed_quad_1d([](double x) { return x; },
                                                 builder_1d(GaussFamily::Legendre, 0, {2.0, 5.0})));

    EXPECT_RELATIVE_EQ(integral, 10.5, MAX_RELATIVE_ERROR);
}

TEST(Fixed, Fixed1dIntegratesPolynomialDataset)
{
    const auto data = data_1d();
    const auto range = data.range;
    ASSERT_FALSE(data.polynomials.empty());

    for (const GaussFamily family : {GaussFamily::Legendre, GaussFamily::Lobatto})
    {
        for (const auto& polynomial : data.polynomials)
        {
            SCOPED_TRACE(std::format("{} {}", family, polynomial.name));
            const std::size_t exactness = polynomial.coeffs.size() - 1;
            const auto evaluate = [&](double x)
            { return evaluate_polynomial(polynomial.coeffs, x); };

            const double integral =
                unwrap(fixed_quad_1d(evaluate, builder_1d(family, exactness, range)));
            EXPECT_RELATIVE_EQ(polynomial.integral, integral, MAX_RELATIVE_ERROR);

            const double width = (range[1] - range[0]) / 3.0;
            auto subdivided = unwrap(builder_1d(family, exactness, range)
                                         .subdivisions({range[0] + width, range[0] + 2.0 * width}));
            const double subdivided_integral =
                unwrap(fixed_quad_1d(evaluate, std::move(subdivided)));
            EXPECT_RELATIVE_EQ(polynomial.integral, subdivided_integral, MAX_RELATIVE_ERROR);
        }
    }
}
//}}}

//{{{ collection: two-dimensional tests
TEST(Fixed, Fixed2dIntegratesPolynomialDataset)
{
    const auto data = data_2d();
    const auto range = data.range;
    ASSERT_FALSE(data.polynomials.empty());

    for (const GaussFamily family : {GaussFamily::Legendre, GaussFamily::Lobatto})
    {
        for (const auto& polynomial : data.polynomials)
        {
            SCOPED_TRACE(std::format("{} {}", family, polynomial.name));
            const std::size_t u_exactness = polynomial.coeffs_u.size() - 1;
            const std::size_t v_exactness = polynomial.coeffs_v.size() - 1;
            const auto evaluate = [&](double u, double v)
            {
                return evaluate_polynomial(polynomial.coeffs_u, u) *
                       evaluate_polynomial(polynomial.coeffs_v, v);
            };

            const double integral = unwrap(
                fixed_quad_2d(evaluate, builder_2d(family, u_exactness, v_exactness, range)));
            EXPECT_RELATIVE_EQ(polynomial.integral, integral, MAX_RELATIVE_ERROR);

            const double u_width = (range[1] - range[0]) / 3.0;
            const double v_width = (range[3] - range[2]) / 3.0;
            auto subdivided =
                unwrap(builder_2d(family, u_exactness, v_exactness, range)
                           .subdivisions({range[0] + u_width, range[0] + 2.0 * u_width},
                                         {range[2] + v_width, range[2] + 2.0 * v_width}));
            const double subdivided_integral =
                unwrap(fixed_quad_2d(evaluate, std::move(subdivided)));
            EXPECT_RELATIVE_EQ(polynomial.integral, subdivided_integral, MAX_RELATIVE_ERROR);
        }
    }
}
//}}}
