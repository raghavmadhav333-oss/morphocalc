#pragma once

#include <array>
#include <vector>
#include <cmath>
#include <algorithm>

namespace morphocalc {

/**
 * @brief Velocity gradient tensor at a 3D continuum point.
 */
struct VelocityGradient {
    float dudx, dudy, dudz;
    float dvdx, dvdy, dvdz;
    float dwdx, dwdy, dwdz;
};

/**
 * @brief Symmetric strain-rate tensor S_ij = 0.5 * (du_i/dx_j + du_j/du_i)
 */
struct StrainRateTensor {
    float S11, S12, S13;
    float S21, S22, S23;
    float S31, S32, S33;

    float magnitude() const {
        return std::sqrt(2.0f * (S11*S11 + S22*S22 + S33*S33 +
                                 2.0f * (S12*S12 + S13*S13 + S23*S23)));
    }
};

/**
 * @brief Anti-symmetric rotation tensor Omega_ij = 0.5 * (du_i/dx_j - du_j/du_i)
 */
struct RotationTensor {
    float W11, W12, W13;
    float W21, W22, W23;
    float W31, W32, W33;

    float magnitude() const {
        return std::sqrt(2.0f * (W12*W12 + W13*W13 + W23*W23));
    }
};

/**
 * @brief Lumley turbulence triangle invariants (II_b, III_b) and Hunt Q-criterion.
 */
struct TopologicalInvariants {
    float Q_criterion;
    float R_criterion;
    float II_b;
    float III_b;
    float enstrophy;
    float dissipation;
};

/**
 * @brief Pope 10-tensor integrity basis for turbulence modeling (S.B. Pope 1975).
 * Represents all objective, frame-indifferent tensor combinations up to 5th order.
 */
struct PopeTensorBasis {
    std::array<std::array<float, 9>, 10> T; // 10 tensors, each 3x3 (flattened to 9 floats)
};

class InvariantEngine {
public:
    static StrainRateTensor compute_strain_rate(const VelocityGradient& grad);
    static RotationTensor compute_rotation_tensor(const VelocityGradient& grad);
    static TopologicalInvariants compute_topological_invariants(const VelocityGradient& grad);
    static PopeTensorBasis compute_pope_basis(const StrainRateTensor& S, const RotationTensor& W);
    
    // Batch vectorized computation over spatial mesh
    static void compute_mesh_invariants(
        const std::vector<VelocityGradient>& grads,
        std::vector<TopologicalInvariants>& out_invariants,
        std::vector<PopeTensorBasis>& out_bases
    );
};

} // namespace morphocalc
