/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
  \\    /   O peration     |
   \\  /    A nd           | Copyright (C) 2026 MorphoCalc-Nexus Research Initiative
    \\/     M anipulation  |
-------------------------------------------------------------------------------
Class
    Foam::LESModels::morphoCalcCompressibleLES

Description
    Production OpenFOAM Drop-In Compressible LES Plugin with Acoustic Stiffness
    Decoupling, SO(3) Frame-Indifferent Lie-Algebra Decomposition, and
    Ducros-Mach Compressive Shock Sensor Gate.

    Features:
      - Traceless deviatoric strain-rate S^d = S - (1/3)*theta*I
      - Orthogonal Lie decomposition: L = S^d + Omega + (1/3)*theta*I
      - Solenoidal vs. acoustic 4-stage Runge-Kutta curvature splitting
      - Ducros-Mach shock gate Xi in [0, 1) protecting boundary layer vortices
      - Zero global MPI_Allreduce collective operations (strictly local SIMD)
\*---------------------------------------------------------------------------*/

#ifndef morphoCalcCompressibleLES_H
#define morphoCalcCompressibleLES_H

#include "fvCFD.H"
#include "turbulentFluidThermoModel.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "symmTensorField.H"
#include "tensorField.H"

namespace Foam
{
namespace LESModels
{

//- Rational Padé denominator function with overflow protection
inline scalar padeDenominator(const scalar D) noexcept;

//- Decoupled compressible MorphoCalc closure computation
void computeDecoupledMorphoCalcClosure
(
    const volVectorField& U,
    const volScalarField& rho,
    const volScalarField& p,
    const volScalarField& thermoPsi,
    const scalarField& deltaVol,
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
);

} // End namespace LESModels
} // End namespace Foam

#endif
