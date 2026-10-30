#include "tensor_invariants.hpp"

namespace morphocalc {

StrainRateTensor InvariantEngine::compute_strain_rate(const VelocityGradient& g) {
    StrainRateTensor s;
    s.S11 = g.dudx;
    s.S22 = g.dvdy;
    s.S33 = g.dwdz;

    s.S12 = 0.5f * (g.dudy + g.dvdx);
    s.S21 = s.S12;

    s.S13 = 0.5f * (g.dudz + g.dwdx);
    s.S31 = s.S13;

    s.S23 = 0.5f * (g.dvdz + g.dwdy);
    s.S32 = s.S23;
    return s;
}

RotationTensor InvariantEngine::compute_rotation_tensor(const VelocityGradient& g) {
    RotationTensor w;
    w.W11 = 0.0f;
    w.W22 = 0.0f;
    w.W33 = 0.0f;

    w.W12 = 0.5f * (g.dudy - g.dvdx);
    w.W21 = -w.W12;

    w.W13 = 0.5f * (g.dudz - g.dwdx);
    w.W31 = -w.W13;

    w.W23 = 0.5f * (g.dvdz - g.dwdy);
    w.W32 = -w.W23;
    return w;
}

TopologicalInvariants InvariantEngine::compute_topological_invariants(const VelocityGradient& g) {
    StrainRateTensor s = compute_strain_rate(g);
    RotationTensor w = compute_rotation_tensor(g);

    float s_sq = s.S11*s.S11 + s.S22*s.S22 + s.S33*s.S33 +
                 2.0f * (s.S12*s.S12 + s.S13*s.S13 + s.S23*s.S23);

    float w_sq = 2.0f * (w.W12*w.W12 + w.W13*w.W13 + w.W23*w.W23);

    TopologicalInvariants inv;
    // Hunt Q-criterion: Q = 0.5 * (||W||^2 - ||S||^2)
    inv.Q_criterion = 0.5f * (w_sq - s_sq);
    inv.enstrophy = 0.5f * w_sq;
    inv.dissipation = s_sq;

    // Second invariant of anisotropy tensor b_ij
    inv.II_b = -0.5f * s_sq / (s_sq + 1e-8f);
    // Determinant approximation for Lumley III_b
    inv.III_b = (s.S11*(s.S22*s.S33 - s.S23*s.S23) -
                 s.S12*(s.S21*s.S33 - s.S23*s.S31) +
                 s.S13*(s.S21*s.S32 - s.S22*s.S31)) / (std::pow(s_sq + 1e-8f, 1.5f));
    inv.R_criterion = -inv.III_b;

    return inv;
}

// 3x3 matrix multiplication helper
inline void mat_mul_3x3(const float* A, const float* B, float* C) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            C[i*3 + j] = A[i*3 + 0] * B[0*3 + j] +
                         A[i*3 + 1] * B[1*3 + j] +
                         A[i*3 + 2] * B[2*3 + j];
        }
    }
}

// Subtract trace / 3 * I (deviatoric projection)
inline void make_deviatoric(float* A) {
    float tr = (A[0] + A[4] + A[8]) / 3.0f;
    A[0] -= tr;
    A[4] -= tr;
    A[8] -= tr;
}

PopeTensorBasis InvariantEngine::compute_pope_basis(const StrainRateTensor& s_in, const RotationTensor& w_in) {
    PopeTensorBasis basis;

    float S[9] = { s_in.S11, s_in.S12, s_in.S13,
                   s_in.S21, s_in.S22, s_in.S23,
                   s_in.S31, s_in.S32, s_in.S33 };

    float W[9] = { w_in.W11, w_in.W12, w_in.W13,
                   w_in.W21, w_in.W22, w_in.W23,
                   w_in.W31, w_in.W32, w_in.W33 };

    float S2[9], W2[9], SW[9], WS[9];
    mat_mul_3x3(S, S, S2);
    mat_mul_3x3(W, W, W2);
    mat_mul_3x3(S, W, SW);
    mat_mul_3x3(W, S, WS);

    // T1 = S
    for (int k = 0; k < 9; ++k) basis.T[0][k] = S[k];

    // T2 = S*W - W*S
    for (int k = 0; k < 9; ++k) basis.T[1][k] = SW[k] - WS[k];

    // T3 = S^2 - 1/3 tr(S^2) I
    for (int k = 0; k < 9; ++k) basis.T[2][k] = S2[k];
    make_deviatoric(basis.T[2].data());

    // T4 = W^2 - 1/3 tr(W^2) I
    for (int k = 0; k < 9; ++k) basis.T[3][k] = W2[k];
    make_deviatoric(basis.T[3].data());

    // T5 = W*S^2 - S^2*W
    float WS2[9], S2W[9];
    mat_mul_3x3(W, S2, WS2);
    mat_mul_3x3(S2, W, S2W);
    for (int k = 0; k < 9; ++k) basis.T[4][k] = WS2[k] - S2W[k];

    // T6 = W^2*S + S*W^2 - 2/3 tr(S*W^2) I
    float W2S[9], SW2[9];
    mat_mul_3x3(W2, S, W2S);
    mat_mul_3x3(S, W2, SW2);
    for (int k = 0; k < 9; ++k) basis.T[5][k] = W2S[k] + SW2[k];
    make_deviatoric(basis.T[5].data());

    // T7 = W*S*W^2 - W^2*S*W
    float WSW2[9], W2SW[9];
    mat_mul_3x3(WS, W2, WSW2);
    mat_mul_3x3(W2S, W, W2SW);
    for (int k = 0; k < 9; ++k) basis.T[6][k] = WSW2[k] - W2SW[k];

    // T8 = S*W*S^2 - S^2*W*S
    float SWS2[9], S2WS[9];
    mat_mul_3x3(SW, S2, SWS2);
    mat_mul_3x3(S2, WS, S2WS);
    for (int k = 0; k < 9; ++k) basis.T[7][k] = SWS2[k] - S2WS[k];

    // T9 = W^2*S^2 + S^2*W^2 - 2/3 tr(S^2*W^2) I
    float W2S2[9], S2W2[9];
    mat_mul_3x3(W2, S2, W2S2);
    mat_mul_3x3(S2, W2, S2W2);
    for (int k = 0; k < 9; ++k) basis.T[8][k] = W2S2[k] + S2W2[k];
    make_deviatoric(basis.T[8].data());

    // T10 = W*S^2*W^2 - W^2*S^2*W
    float WS2W2[9], W2S2W[9];
    mat_mul_3x3(WS2, W2, WS2W2);
    mat_mul_3x3(W2, S2W, W2S2W);
    for (int k = 0; k < 9; ++k) basis.T[9][k] = WS2W2[k] - W2S2W[k];

    return basis;
}

void InvariantEngine::compute_mesh_invariants(
    const std::vector<VelocityGradient>& grads,
    std::vector<TopologicalInvariants>& out_invariants,
    std::vector<PopeTensorBasis>& out_bases
) {
    const size_t n = grads.size();
    out_invariants.resize(n);
    out_bases.resize(n);

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        StrainRateTensor s = compute_strain_rate(grads[i]);
        RotationTensor w = compute_rotation_tensor(grads[i]);
        out_invariants[i] = compute_topological_invariants(grads[i]);
        out_bases[i] = compute_pope_basis(s, w);
    }
}

} // namespace morphocalc
