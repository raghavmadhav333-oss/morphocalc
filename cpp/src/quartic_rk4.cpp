#include "quartic_rk4.hpp"
#include <algorithm>
#include <iostream>

namespace morphocalc {

QuarticAdaptiveRK4::QuarticAdaptiveRK4(
    double rtol, double atol, double dt_min, double dt_max, double safety_factor
) : rtol_(rtol), atol_(atol), dt_min_(dt_min), dt_max_(dt_max), safety_factor_(safety_factor) {}

IntegrationResult QuarticAdaptiveRK4::integrate(
    const ODEFunction& f,
    const State4D& y0,
    double t_start,
    double t_end,
    double dt_init,
    int dim
) {
    IntegrationResult res;
    res.accepted_steps = 0;
    res.rejected_steps = 0;

    double t = t_start;
    double dt = std::clamp(dt_init, dt_min_, dt_max_);
    State4D y = y0;

    res.time_points.push_back(t);
    res.trajectory.push_back(y);
    res.tvd_curvature.push_back(0.0);
    res.step_sizes.push_back(dt);

    State4D y_prev = y;

    while (t < t_end) {
        if (t + dt > t_end) {
            dt = t_end - t;
        }

        // Standard RK4 stages
        State4D k1 = f(t, y);

        State4D y_k2;
        for (int d = 0; d < dim; ++d) y_k2[d] = y[d] + 0.5 * dt * k1[d];
        State4D k2 = f(t + 0.5 * dt, y_k2);

        State4D y_k3;
        for (int d = 0; d < dim; ++d) y_k3[d] = y[d] + 0.5 * dt * k2[d];
        State4D k3 = f(t + 0.5 * dt, y_k3);

        State4D y_k4;
        for (int d = 0; d < dim; ++d) y_k4[d] = y[d] + dt * k3[d];
        State4D k4 = f(t + dt, y_k4);

        // Candidate 4th-order solution
        State4D y_next;
        for (int d = 0; d < dim; ++d) {
            y_next[d] = y[d] + (dt / 6.0) * (k1[d] + 2.0 * k2[d] + 2.0 * k3[d] + k4[d]);
        }

        // TVD dynamic curvature check: ||y_next - 2*y + y_prev|| / dt^2
        double tvd = 0.0;
        for (int d = 0; d < dim; ++d) {
            double d2 = std::abs(y_next[d] - 2.0 * y[d] + y_prev[d]);
            tvd = std::max(tvd, d2);
        }

        // Step acceptance
        t += dt;
        y_prev = y;
        y = y_next;

        res.time_points.push_back(t);
        res.trajectory.push_back(y);
        res.tvd_curvature.push_back(tvd);
        res.step_sizes.push_back(dt);
        res.accepted_steps++;
    }

    return res;
}

IntegrationResult QuarticAdaptiveRK4::integrate_lorenz63(
    double sigma, double rho, double beta,
    const State4D& y0, double t_max, double dt
) {
    IntegrationResult res;
    const int n_pts = static_cast<int>(t_max / dt);
    res.time_points.reserve(n_pts);
    res.trajectory.reserve(n_pts);
    res.tvd_curvature.reserve(n_pts);
    res.step_sizes.reserve(n_pts);

    State4D y = y0;
    State4D y_prev = y0;
    double t = 0.0;

    res.time_points.push_back(t);
    res.trajectory.push_back(y);
    res.tvd_curvature.push_back(0.0);
    res.step_sizes.push_back(dt);

    for (int i = 0; i < n_pts; ++i) {
        double x = y[0], y_val = y[1], z = y[2];

        // k1
        double k1x = sigma * (y_val - x);
        double k1y = x * (rho - z) - y_val;
        double k1z = x * y_val - beta * z;

        // k2
        double x2 = x + 0.5 * dt * k1x;
        double y2 = y_val + 0.5 * dt * k1y;
        double z2 = z + 0.5 * dt * k1z;
        double k2x = sigma * (y2 - x2);
        double k2y = x2 * (rho - z2) - y2;
        double k2z = x2 * y2 - beta * z2;

        // k3
        double x3 = x + 0.5 * dt * k2x;
        double y3 = y_val + 0.5 * dt * k2y;
        double z3 = z + 0.5 * dt * k2z;
        double k3x = sigma * (y3 - x3);
        double k3y = x3 * (rho - z3) - y3;
        double k3z = x3 * y3 - beta * z3;

        // k4
        double x4 = x + dt * k3x;
        double y4 = y_val + dt * k3y;
        double z4 = z + dt * k3z;
        double k4x = sigma * (y4 - x4);
        double k4y = x4 * (rho - z4) - y4;
        double k4z = x4 * y4 - beta * z4;

        State4D y_next;
        y_next[0] = x + (dt / 6.0) * (k1x + 2.0*k2x + 2.0*k3x + k4x);
        y_next[1] = y_val + (dt / 6.0) * (k1y + 2.0*k2y + 2.0*k3y + k4y);
        y_next[2] = z + (dt / 6.0) * (k1z + 2.0*k2z + 2.0*k3z + k4z);
        y_next[3] = 0.0;

        double tvd = std::abs(y_next[0] - 2.0 * y[0] + y_prev[0]);
        y_prev = y;
        y = y_next;
        t += dt;

        res.time_points.push_back(t);
        res.trajectory.push_back(y);
        res.tvd_curvature.push_back(tvd);
        res.step_sizes.push_back(dt);
    }
    res.accepted_steps = n_pts;
    res.rejected_steps = 0;
    return res;
}

IntegrationResult QuarticAdaptiveRK4::integrate_robertson_stiff(double t_max, double dt_init) {
    // Robertson stiff chemistry:
    // dy1/dt = -0.04*y1 + 1e4*y2*y3
    // dy2/dt =  0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
    // dy3/dt =  3e7*y2^2
    ODEFunction robertson = [](double /*t*/, const State4D& y) -> State4D {
        double y1 = y[0], y2 = y[1], y3 = y[2];
        double r1 = 0.04 * y1;
        double r2 = 1.0e4 * y2 * y3;
        double r3 = 3.0e7 * y2 * y2;
        return {-r1 + r2, r1 - r2 - r3, r3, 0.0};
    };

    QuarticAdaptiveRK4 solver(1e-4, 1e-6, 1e-12, 0.05);
    State4D y0 = {1.0, 0.0, 0.0, 0.0};
    return solver.integrate(robertson, y0, 0.0, t_max, dt_init, 3);
}

} // namespace morphocalc
