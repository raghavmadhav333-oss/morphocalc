#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor_invariants.hpp"
#include "quartic_rk4.hpp"
#include "symbolic_hunter.hpp"

namespace py = pybind11;
using namespace morphocalc;

PYBIND11_MODULE(morphocalc_cpp, m) {
    m.doc() = "MorphoCalc-Core: High-Performance C++20 Physics Acceleration Engine";

    // 1. Quartic RK4 ODE Integrator
    py::class_<IntegrationResult>(m, "IntegrationResult")
        .def_readonly("time_points", &IntegrationResult::time_points)
        .def_readonly("trajectory", &IntegrationResult::trajectory)
        .def_readonly("tvd_curvature", &IntegrationResult::tvd_curvature)
        .def_readonly("step_sizes", &IntegrationResult::step_sizes)
        .def_readonly("accepted_steps", &IntegrationResult::accepted_steps);

    py::class_<QuarticAdaptiveRK4>(m, "QuarticAdaptiveRK4")
        .def(py::init<double, double, double, double, double>(),
             py::arg("rtol") = 1e-6, py::arg("atol") = 1e-8,
             py::arg("dt_min") = 1e-9, py::arg("dt_max") = 0.1,
             py::arg("safety_factor") = 0.85)
        .def_static("integrate_lorenz63", &QuarticAdaptiveRK4::integrate_lorenz63,
                    py::arg("sigma") = 10.0, py::arg("rho") = 28.0, py::arg("beta") = 8.0/3.0,
                    py::arg("y0") = State4D{1.0, 1.0, 1.0, 0.0},
                    py::arg("t_max") = 20.0, py::arg("dt") = 0.01)
        .def_static("integrate_robertson_stiff", &QuarticAdaptiveRK4::integrate_robertson_stiff,
                    py::arg("t_max") = 10.0, py::arg("dt_init") = 1e-4);

    // 2. S.B. Pope Tensor & Topological Invariants
    py::class_<TopologicalInvariants>(m, "TopologicalInvariants")
        .def_readonly("Q_criterion", &TopologicalInvariants::Q_criterion)
        .def_readonly("R_criterion", &TopologicalInvariants::R_criterion)
        .def_readonly("II_b", &TopologicalInvariants::II_b)
        .def_readonly("III_b", &TopologicalInvariants::III_b)
        .def_readonly("enstrophy", &TopologicalInvariants::enstrophy)
        .def_readonly("dissipation", &TopologicalInvariants::dissipation);

    // 3. Symbolic Law Hunter
    py::class_<CandidateLaw>(m, "CandidateLaw")
        .def_readonly("formula", &CandidateLaw::formula)
        .def_readonly("mse_train", &CandidateLaw::mse_train)
        .def_readonly("second_law_consistency", &CandidateLaw::second_law_consistency)
        .def_readonly("complexity", &CandidateLaw::complexity)
        .def_readonly("pareto_score", &CandidateLaw::pareto_score);

    py::class_<SymbolicLawHunter>(m, "SymbolicLawHunter")
        .def_static("discover_closure_laws", &SymbolicLawHunter::discover_closure_laws,
                    py::arg("S_mag"), py::arg("Omega_mag"), py::arg("S12"),
                    py::arg("target_tau"), py::arg("max_complexity") = 5);
}
#endif
