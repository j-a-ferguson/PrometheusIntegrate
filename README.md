# PrometheusIntegrate (C++23)

A C++23 port of [`PrometheusIntegrate`](https://github.com/PrometheusLabs/PrometheusIntegrate)
(v0.1.1, `dev` branch): validated Gaussian fixed and adaptive quadrature in one and two
dimensions. The public API, validation rules, error messages and algorithms follow the Rust
crate one-to-one; Armadillo replaces `prometheus-linalg`, and the optional `trace` and `serde`
features are not ported.

## Build

Requires CMake ≥ 3.25, Ninja, a C++23 compiler with `<expected>` (GCC ≥ 12, Clang ≥ 19 with
libstdc++, or any Clang/AppleClang using libc++ ≥ 17), and vcpkg (`VCPKG_ROOT` set).

```sh
cmake --preset release        # or: debug, asan (ASan + UBSan)
cmake --build --preset release
ctest --preset release
```

Dependencies come from `vcpkg.json`: `armadillo` for the library; `gtest` and `nlohmann-json`
only under the `tests` manifest feature, which CMake enables automatically when
`PROMETHEUS_INTEGRATE_BUILD_TESTS=ON` (the default). On Linux, vcpkg builds LAPACK from
`lapack-reference`, which needs a Fortran compiler (`gfortran`); on macOS it uses Accelerate.

`find_package(Armadillo)` also accepts a system Armadillo (CMake's `FindArmadillo`), so the
project builds without vcpkg as long as Armadillo, GoogleTest and nlohmann-json are findable.

CMake options: `PROMETHEUS_INTEGRATE_BUILD_TESTS` (ON), `PROMETHEUS_INTEGRATE_SANITIZE` (OFF),
`PROMETHEUS_INTEGRATE_WERROR` (OFF). Everything is compiled with `-ffp-contract=off` so the
compiler never fuses `a*b + c` into an FMA, matching Rust's floating-point semantics.

## Layout

| Rust                     | C++                                                     |
|--------------------------|---------------------------------------------------------|
| `src/lib.rs` re-exports  | `include/prometheus/integrate.hpp` (aliases + forwarders) |
| `src/config.rs`          | `include/.../config.hpp`, `src/config.cpp`              |
| `src/gauss.rs`           | `include/.../gauss.hpp`, `src/gauss.cpp` (only file using Armadillo) |
| `src/integration.rs`     | `include/.../integration.hpp`, `src/integration.cpp`    |
| `src/common.rs`          | `include/.../common.hpp`                                |
| `src/fixed/{d1,d2}.rs`   | `include/.../fixed/{d1,d2}.hpp`, `src/fixed_{d1,d2}.cpp` |
| `src/adaptive/{d1,d2}.rs`| `include/.../adaptive/{d1,d2}.hpp`, `src/adaptive_{d1,d2}.cpp` |
| `tests/*.rs`             | `tests/*_test.cpp` (gtest)                              |
| in-module `#[cfg(test)]` | `tests/internal_test.cpp` (via `detail::`)              |

Integrand-generic code (`integrate`, `integrate_over`, the adaptive loops) is templated and
therefore lives in the headers; everything else is in `src/`.

## Rust → C++ mapping

| Rust | C++ |
|---|---|
| data-carrying `enum` + `match` (`ConfigIssue`, `RuleError`, `IntegrationError`, `EvaluationPoint`) | `std::variant` of plain structs in a namespace named after the enum (`config_issue::InvalidTolerance`, …) + `std::visit` with an `overloaded{…}` lambda set |
| unit-only `enum` (`GaussFamily`, `RuleAxis`) | `enum class` + `switch` |
| `OptionsError::{Rule, Config, Integration}` (`#[from]`) | `std::variant<RuleError, ConfigError, IntegrationError>`: the alternative's type is the tag |
| `Result<T, E>` / `?` | `std::expected<T, E>`; early `return std::unexpected{…}`, or `and_then` / `transform_error` (see `adaptive_quad`) |
| `Option<T>` | `std::optional<T>` |
| `Result<&'a T, E>` | `std::expected<std::reference_wrapper<const T>, E>` (alias `Ref<T>`) |
| `Type::new(..)` | `Type::create(..)` (`new` is a keyword); infallible `new` → constructor |
| `pub(crate) new_unchecked` | public `new_unchecked` (C++ has no crate-private visibility) |
| consuming builder (`fn f(self) -> Self`) | `&&`-qualified member functions: call on a temporary or `std::move` |
| `I: IntoIterator<Item = f64>` | any `std::ranges::input_range` convertible to `double`, plus an `std::initializer_list<double>` overload |
| `F: FnMut(f64) -> f64` | `Integrand1d` / `Integrand2d` concepts; always invoked as an lvalue, so stateful lambdas work |
| `OnceLock` cache | function-local `static` (thread-safe initialisation) |
| `Display` / `to_string()` | `std::formatter` specialisations + `prometheus::integrate::to_string`; floats print exactly as Rust does (`0.00000001`, `NaN`, `inf`) |
| `Default` | default constructor (only `RefinementDepth`, `AxisDepths`) |
| `Hash` | `std::hash` specialisations |
| `len` / `is_empty` / `iter` | `size` / `empty` / `begin`/`end` |
| `#[repr(C)]` nodes | standard-layout classes, size asserted in tests |

## Numerical agreement with the Rust crate

All 65 ported tests pass at the original Rust tolerances (including the `1e-14` relative bound
on the polynomial datasets). A side-by-side harness running both implementations gave:

* **Gauss rules (102 rules, 2703 point/weight pairs):** 51 rules bit-identical; the rest differ
  by at most 2.3e-15 absolute. The two LAPACK eigensolver paths differ in the last ulp.
* **Fixed polynomial integrals (64 cases, plain and subdivided):** at most 5.8e-16 relative
  difference; identical error against the analytic values.
* **Adaptive (11 cases: `sin`, `exp`, `√x`, `ln x`, Runge, oscillatory, step, Gaussian bump,
  disc, `√(u²+v²)`, …):** bit-identical integrals, error estimates, region counts and
  evaluation counts.

Rust's `f64::powi` is replicated in the tests (`test_support::powi`) because
`std::pow(double, int)` rounds differently in the last bit.
