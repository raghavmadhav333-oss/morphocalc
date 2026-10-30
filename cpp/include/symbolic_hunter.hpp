#pragma once

#include <vector>
#include <string>
#include <array>
#include <cmath>
#include <algorithm>

namespace morphocalc {

struct CandidateLaw {
    std::string formula;
    float mse_train;
    float mse_test;
    float second_law_consistency; // Percentage (0 - 100%)
    int complexity;
    float pareto_score;
};

class SymbolicLawHunter {
public:
    /**
     * @brief High-speed combinatorial bitmask search over candidate algebraic closure equations.
     * Evaluates millions of tensor invariant contractions per second.
     */
    static std::vector<CandidateLaw> discover_closure_laws(
        const std::vector<float>& S_mag,
        const std::vector<float>& Omega_mag,
        const std::vector<float>& S12,
        const std::vector<float>& target_tau,
        int max_complexity = 5
    );
};

} // namespace morphocalc
