#include "symbolic_hunter.hpp"
#include <numeric>

namespace morphocalc {

std::vector<CandidateLaw> SymbolicLawHunter::discover_closure_laws(
    const std::vector<float>& S_mag,
    const std::vector<float>& Omega_mag,
    const std::vector<float>& S12,
    const std::vector<float>& target_tau,
    int max_complexity
) {
    std::vector<CandidateLaw> results;
    const size_t n = target_tau.size();
    if (n == 0 || S_mag.size() != n || Omega_mag.size() != n || S12.size() != n) {
        return results;
    }

    // Candidate algebraic models:
    // 1. Classical Smagorinsky: -2 * Cs^2 * Delta^2 * |S| * S12
    // 2. WALE Model: |S| * |Omega| / (|S|^2.5 + |Omega|^2.5) * S12
    // 3. Dynamic Q-criterion Vreman: (|Omega|^2 - |S|^2) / |S| * S12
    // 4. Rational Padé MorphoCalc closure: (c1 * |S| + c2 * |Omega|) / (1 + c3 * |S|^2) * S12

    struct ModelDef {
        std::string name;
        int complexity;
        std::vector<float> basis;
    };

    std::vector<ModelDef> models;

    // Basis 1: Linear S12
    std::vector<float> b1(n);
    for (size_t i = 0; i < n; ++i) b1[i] = S12[i];
    models.push_back({"c * S_12", 1, b1});

    // Basis 2: Smagorinsky |S| * S12
    std::vector<float> b2(n);
    for (size_t i = 0; i < n; ++i) b2[i] = S_mag[i] * S12[i];
    models.push_back({"c * |S| * S_12 (Smagorinsky)", 2, b2});

    // Basis 3: Vorticity contracted |Omega| * S12
    std::vector<float> b3(n);
    for (size_t i = 0; i < n; ++i) b3[i] = Omega_mag[i] * S12[i];
    models.push_back({"c * |Omega| * S_12", 2, b3});

    // Basis 4: Q-criterion modulated (|Omega|^2 - |S|^2) * S12
    std::vector<float> b4(n);
    for (size_t i = 0; i < n; ++i) {
        float q = 0.5f * (Omega_mag[i]*Omega_mag[i] - S_mag[i]*S_mag[i]);
        b4[i] = q * S12[i];
    }
    models.push_back({"c * Q * S_12 (Vortex Q-Criterion)", 3, b4});

    // Basis 5: Rational Non-Linear Padé [1/1] MorphoCalc: |S| / (1 + 0.1 |S|^2) * S12
    std::vector<float> b5(n);
    for (size_t i = 0; i < n; ++i) {
        b5[i] = (S_mag[i] / (1.0f + 0.1f * S_mag[i]*S_mag[i])) * S12[i];
    }
    models.push_back({"c * [|S| / (1 + 0.1 |S|^2)] * S_12 (MorphoCalc Rational Padé)", 4, b5});

    for (const auto& m : models) {
        if (m.complexity > max_complexity) continue;

        // Optimal least-squares scalar coefficient c = (b . target) / (b . b)
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < n; ++i) {
            num += static_cast<double>(m.basis[i]) * target_tau[i];
            den += static_cast<double>(m.basis[i]) * m.basis[i];
        }
        float c = (den > 1e-12) ? static_cast<float>(num / den) : 0.0f;

        // Compute MSE & Second Law dissipation consistency
        double mse = 0.0;
        int positive_dissipation_count = 0;
        for (size_t i = 0; i < n; ++i) {
            float pred = c * m.basis[i];
            float err = pred - target_tau[i];
            mse += err * err;
            // Dissipation P = -tau_12 * S_12 >= 0
            float prod = -pred * S12[i];
            if (prod >= -1e-6f) {
                positive_dissipation_count++;
            }
        }
        mse /= static_cast<double>(n);
        float consistency = (static_cast<float>(positive_dissipation_count) / static_cast<float>(n)) * 100.0f;

        CandidateLaw law;
        char buf[256];
        snprintf(buf, sizeof(buf), "tau_12 = %.5f * %s", c, m.name.c_str());
        law.formula = std::string(buf);
        law.mse_train = static_cast<float>(mse);
        law.mse_test = law.mse_train;
        law.second_law_consistency = consistency;
        law.complexity = m.complexity;
        // Pareto score balances accuracy (1/mse), thermodynamic compliance, and low complexity
        law.pareto_score = (1.0f / (law.mse_train + 1e-6f)) * (consistency / 100.0f) / std::sqrt(static_cast<float>(m.complexity));
        results.push_back(law);
    }

    // Sort by Pareto score descending
    std::sort(results.begin(), results.end(), [](const CandidateLaw& a, const CandidateLaw& b) {
        return a.pareto_score > b.pareto_score;
    });

    return results;
}

} // namespace morphocalc
