// File: morphoCalcCompressibleLES.C
// Target: OpenFOAM v2312 / OpenFOAM-11 Compressible LES Plugin
// Zero global MPI_Allreduce calls; strictly local SIMD-vectorizable cell loop.

#include "morphoCalcCompressibleLES.H"

namespace Foam
{
namespace LESModels
{

inline scalar padeDenominator(const scalar D) noexcept
{
    // C-infinity algebraically identical branches with Float64 overflow guard at D = 1e6
    if (D <= 0.0) return 1.0;
    if (D <= 1.0e6)
    {
        return 1.0 + (0.05 * D * D) / (1.0 + 0.005 * D);
    }
    return 1.0 + (10.0 * D) / (1.0 + 200.0 / D);
}

void computeDecoupledMorphoCalcClosure
(
    const volVectorField& U,
    const volScalarField& rho,
    const volScalarField& p,
    const volScalarField& thermoPsi, // psi = 1/(R*T) = rho/p
    const scalarField& deltaVol,     // cell filter width Delta = V^(1/3)
    const symmTensorField& Sd_stage1,
    const symmTensorField& Sd_stage2,
    const symmTensorField& Sd_stage3,
    const symmTensorField& Sd_stage4,
    const tensorField& Om_stage1,
    const tensorField& Om_stage2,
    const tensorField& Om_stage3,
    const tensorField& Om_stage4,
    const scalarField& theta_stage1,
    const scalarField& theta_stage2,
    const scalarField& theta_stage3,
    const scalarField& theta_stage4,
    scalarField& denSolenoidalOut,
    scalarField& denAcousticOut,
    symmTensorField& tauSGSPadeOut
)
{
    const scalar cs  = 0.0289; // (0.17)^2 Smagorinsky constant
    const scalar cp  = 0.0540; // Pope dispersive commutator coefficient
    const scalar cd  = 0.1200; // Padé high-strain saturation constant
    const scalar cth = 0.4500; // Dilatational shock bulk-viscosity constant
    const scalar gammaGas = 1.4;
    const scalar eps0 = 1.0e-14;

    const label nCells = U.mesh().nCells();

    for (label celli = 0; celli < nCells; ++celli)
    {
        const scalar Delta = deltaVol[celli];
        const scalar Delta2 = Delta * Delta;
        const scalar aSound = Foam::sqrt(gammaGas / Foam::max(thermoPsi[celli], eps0));
        const scalar tauAc  = Delta / Foam::max(aSound, eps0);

        // 1. 4-Stage Curvature Tensors: C = k4 - k3 - k2 + k1 = O(dt^3)
        const symmTensor CSd =
            Sd_stage4[celli] - Sd_stage3[celli] - Sd_stage2[celli] + Sd_stage1[celli];
        const tensor COm =
            Om_stage4[celli] - Om_stage3[celli] - Om_stage2[celli] + Om_stage1[celli];
        const scalar Cth =
            theta_stage4[celli] - theta_stage3[celli] - theta_stage2[celli] + theta_stage1[celli];

        // 2. Objective SO(3)-Invariant Frobenius Norms: A && B = A_ij B_ij
        const scalar normCSd2 = (CSd && CSd);
        const scalar normCOm2 = (COm && COm);
        const scalar normSd1_2 = (Sd_stage1[celli] && Sd_stage1[celli]);
        const scalar normOm1_2 = (Om_stage1[celli] && Om_stage1[celli]);

        const scalar Dvort =
            (normCSd2 + normCOm2) / (normSd1_2 + normOm1_2 + eps0);

        const scalar th1 = theta_stage1[celli];
        const scalar Dac =
            ((1.0 / 3.0) * Cth * Cth) / ((1.0 / 3.0) * th1 * th1 + eps0);

        // 3. Ducros-Mach Compressive Gate Xi in [0, 1)
        const symmTensor& Sd = Sd_stage4[celli];
        const tensor& Om     = Om_stage4[celli];
        const scalar theta   = theta_stage4[celli];

        const scalar normSd2 = (Sd && Sd);
        const scalar normOm2 = (Om && Om);
        const scalar normSd  = Foam::sqrt(Foam::max(normSd2, 0.0));

        const scalar phiDucros =
            (theta * theta) / (theta * theta + normOm2 + 0.05 * normSd2 + eps0);
        const scalar machGate =
            0.5 * (1.0 - Foam::tanh((Delta * theta) / (0.1 * aSound + eps0)));
        const scalar Xi = phiDucros * machGate;

        // 4. Decoupled Padé Denominators
        const scalar DeffSol = Dvort + (Xi * Xi) * Dac;
        const scalar DeffDil = Dac   + Xi * Dvort;

        denSolenoidalOut[celli] = padeDenominator(DeffSol);
        denAcousticOut[celli]   = padeDenominator(DeffDil);

        // 5. Dual-Basis TBNN-Padé SGS Tensor with Traceless Strain + Dilatational Shock Term
        // Pope Commutator: K = Sd . Om - Om . Sd (symmetric and traceless)
        const symmTensor popeComm = twoSymm(Sd & Om);
        const scalar tau0 = 1.0 / ( Foam::sqrt(normSd2 + normOm2) + 1.0e-10 );

        const scalar denTurb = 1.0 + cd * tau0 * normSd;
        const symmTensor tauDev =
            ((-2.0 * cs * Delta2 * normSd) * Sd + (cp * Delta2) * popeComm) / denTurb;

        const scalar absTheta = Foam::mag(theta);
        const scalar tauBulkScalar =
            (-2.0 / 3.0) * (cth * Delta2 * Xi * absTheta * theta)
            / (1.0 + cd * tauAc * Xi * absTheta);

        tauSGSPadeOut[celli] = tauDev + tauBulkScalar * symmTensor::I;
    }
}

} // End namespace LESModels
} // End namespace Foam
