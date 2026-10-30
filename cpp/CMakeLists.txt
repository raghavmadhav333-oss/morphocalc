cmake_minimum_required(VERSION 3.20)
project(morphocalc_core VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for high-performance physics calculation
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast /openmp)
else()
    add_compile_options(-O3 -mavx2 -mfma -ffast-math -fopenmp -Wall -Wextra)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Static C++ computational library
add_library(morphocalc_static STATIC
    src/tensor_invariants.cpp
    src/quartic_rk4.cpp
    src/symbolic_hunter.cpp
)

# Benchmark / Demo Executable
add_executable(morphocalc_cpp_benchmark
    benchmarks/benchmark_speed.cpp
)
target_link_libraries(morphocalc_cpp_benchmark PRIVATE morphocalc_static)

# Pybind11 Python Module (optional if pybind11 is installed)
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    pybind11_add_module(morphocalc_cpp
        src/pybind_bindings.cpp
    )
    target_link_libraries(morphocalc_cpp PRIVATE morphocalc_static)
    message(STATUS "[✓] Pybind11 detected: building Python C++ extension 'morphocalc_cpp'")
else()
    message(STATUS "[!] Pybind11 not found: building C++ core library and benchmark CLI")
endif()
