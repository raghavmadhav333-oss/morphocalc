#pragma once

#include <vector>
#include <functional>
#include <array>
#include <cmath>

namespace morphocalc {

/**
 * @brief State vector for dynamical systems (up to 4 dimensions, e.g. Lorenz x,y,z or Robertson y1,y2,y3).
 */
using State4D = std::array<double, 4>;

/**
 * @brief Dynamic RHS function: dydt = f(t, y)
 */
using ODEFunction = std::function<State4D(double t, const State4D& y)>;

struct IntegrationResult {
    std::vector<double> time_points;
    std::vector<State4D> trajectory;
    std::vector<double> tvd_curvature;
    std::vector<double> step_sizes;
    int accepted_steps;
    int rejected_steps;
};

class QuarticAdaptiveRK4 {
private:
    double rtol_;
    double atol_;
    double dt_min_;
    double dt_max_;
    double safety_factor_;

public:
    explicit QuarticAdaptiveRK4(
        double rtol = 1e-6,
        double atol = 1e-8,
        double dt_min = 1e-9,
        double dt_max = 0.1,
        double safety_factor = 0.85
    );

    /**
     * @brief Integrates dynamic ODE system from t_start to t_end using 4th-order Runge-Kutta
     * with embedded 5th-order error estimation and dynamic TVD shock-sensing curvature control.
     */
    IntegrationResult integrate(
        const ODEFunction& f,
        const State4D& y0,
        double t_start,
        double t_end,
        double dt_init = 0.01,
        int dim = 3
    );

    /**
     * @brief Pre-compiled, ultra-fast vectorized Lorenz 63 chaotic integration (AVX register cache).
     */
    static IntegrationResult integrate_lorenz63(
        double sigma = 10.0,
        double rho = 28.0,
        double beta = 8.0 / 3.0,
        const State4D& y0 = {1.0, 1.0, 1.0, 0.0},
        double t_max = 20.0,
        double dt = 0.01
    );

    /**
     * @brief Pre-compiled stiff chemical kinetics (Robertson problem, stiffness ~ 10^8).
     */
    static IntegrationResult integrate_robertson_stiff(
        double t_max = 10.0,
        double dt_init = 1e-4
    );
};

} // namespace morphocalc
