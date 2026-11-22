#pragma once
//! Error used by one-shot operations that combine configuration and integration.

//{{{ crate includes
#include "prometheus/integrate/config.hpp"
#include "prometheus/integrate/gauss.hpp"
#include "prometheus/integrate/integration.hpp"
//}}}
//{{{ std includes
#include <variant>
//}}}

namespace prometheus::integrate
{

/// Error returned by an operation that configures and performs integration in one call.
///
/// Rust's `OptionsError::{Rule, Config, Integration}` wrapper variants (with `#[from]`) map to a
/// variant over the three wrapped error types themselves: the alternative's type *is* the tag,
/// and implicit conversion from each error type plays the role of `From`.
using OptionsError = std::variant<RuleError, ConfigError, IntegrationError>;

} // namespace prometheus::integrate

PROMETHEUS_INTEGRATE_DECLARE_FORMATTER(prometheus::integrate::OptionsError);
