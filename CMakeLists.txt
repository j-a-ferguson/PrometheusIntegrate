cmake_minimum_required(VERSION 3.25)

option(PROMETHEUS_INTEGRATE_BUILD_TESTS "Build the GoogleTest suite" ON)
option(PROMETHEUS_INTEGRATE_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(PROMETHEUS_INTEGRATE_WERROR "Treat compiler warnings as errors" OFF)

# Test-only dependencies live behind the vcpkg manifest feature "tests". This must be set
# before project(), which is when the vcpkg toolchain runs `vcpkg install`.
if(PROMETHEUS_INTEGRATE_BUILD_TESTS)
    list(APPEND VCPKG_MANIFEST_FEATURES "tests")
endif()

project(PrometheusIntegrate
    VERSION 0.1.1
    DESCRIPTION "Validated Gaussian fixed and adaptive quadrature in one and two dimensions"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#{{{ dependencies
# Works both with vcpkg (whose wrapper resolves the Armadillo config package and defines the
# `armadillo` target) and with a system install found via CMake's FindArmadillo module.
find_package(Armadillo REQUIRED)
if(NOT TARGET armadillo)
    add_library(armadillo INTERFACE IMPORTED)
    target_include_directories(armadillo SYSTEM INTERFACE ${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(armadillo INTERFACE ${ARMADILLO_LIBRARIES})
endif()
#}}}

#{{{ common compile options
add_library(prometheus_integrate_options INTERFACE)
target_compile_options(prometheus_integrate_options INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
        # Match Rust's floating-point semantics: never fuse a*b+c into an FMA.
        -ffp-contract=off
        $<$<BOOL:${PROMETHEUS_INTEGRATE_WERROR}>:-Werror>>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /fp:precise $<$<BOOL:${PROMETHEUS_INTEGRATE_WERROR}>:/WX>>)
if(PROMETHEUS_INTEGRATE_SANITIZE)
    target_compile_options(prometheus_integrate_options INTERFACE
        -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
    target_link_options(prometheus_integrate_options INTERFACE -fsanitize=address,undefined)
endif()
#}}}

#{{{ library
add_library(prometheus_integrate
    src/adaptive_d1.cpp
    src/adaptive_d2.cpp
    src/config.cpp
    src/display.cpp
    src/fixed_d1.cpp
    src/fixed_d2.cpp
    src/gauss.cpp
    src/integration.cpp)
add_library(prometheus::integrate ALIAS prometheus_integrate)

target_include_directories(prometheus_integrate PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(prometheus_integrate PUBLIC cxx_std_23)
# Armadillo is an implementation detail of gauss.cpp: no public header includes it.
target_link_libraries(prometheus_integrate
    PRIVATE armadillo
    PUBLIC $<BUILD_INTERFACE:prometheus_integrate_options>)
#}}}

#{{{ tests
if(PROMETHEUS_INTEGRATE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
#}}}
