find_package(GTest CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(prometheus_integrate_tests
    adaptive_test.cpp
    config_test.cpp
    fixed_api_test.cpp
    fixed_test.cpp
    gauss_test.cpp
    internal_test.cpp
    traits_test.cpp)

target_link_libraries(prometheus_integrate_tests PRIVATE
    prometheus::integrate
    GTest::gtest_main
    nlohmann_json::nlohmann_json)

# The Rust tests read `assets/*.json` relative to the crate root; here the path is baked in.
target_compile_definitions(prometheus_integrate_tests PRIVATE
    PROMETHEUS_INTEGRATE_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets")

include(GoogleTest)
gtest_discover_tests(prometheus_integrate_tests DISCOVERY_TIMEOUT 60)
