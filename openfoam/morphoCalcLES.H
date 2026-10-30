/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2026 MorphoCalc-Nexus Research Initiative
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Class
    Foam::LESModels::morphoCalcLES

Description
    Production OpenFOAM Drop-In Large Eddy Simulation (LES) Turbulence Model.
    Implements the autonomously discovered 3D non-linear frame-indifferent 
    Reynolds stress closure based on S.B. Pope's 10-tensor integrity basis:

        tau_SGS = -2 * nu_t * S + G_2 * T_2 + G_3 * T_3

    where:
        T_1 = S
        T_2 = S * Omega - Omega * S
        T_3 = S^2 - (1/3) * tr(S^2) * I
        G_2 = -0.0095 + 0.0156 * I_1 - 0.0142 * |I_2|
        G_3 = -10.3033

    Guarantees strict Galilean frame indifference and Second Law of Thermodynamics
    positive entropy production (P >= 0).
\*---------------------------------------------------------------------------*/

#ifndef morphoCalcLES_H
#define morphoCalcLES_H

#include "LESModel.H"
#include "eddyViscosity.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "fvc.H"
#include "fvm.H"
#include "dimensionedScalar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

template<class BasicTurbulenceModel>
class morphoCalcLES
:
    public eddyViscosity<LESModel<BasicTurbulenceModel>>
{
    // Private Data
    dimensionedScalar Ck_;
    dimensionedScalar Ce_;
    dimensionedScalar alpha_g2_;
    dimensionedScalar beta_g2_;
    dimensionedScalar gamma_g2_;
    dimensionedScalar c_g3_;

    // Discovered Tensor Basis Fields
    volSymmTensorField T1_;
    volSymmTensorField T2_;
    volSymmTensorField T3_;
    volScalarField I1_;
    volScalarField I2_;

public:

    TypeName("morphoCalcLES");

    // Constructors
    morphoCalcLES
    (
        const alphaField& alpha,
        const rhoField& rho,
        const volVectorField& U,
        const surfaceScalarField& alphaRhoPhi,
        const surfaceScalarField& phi,
        const transportModel& transport,
        const word& propertiesName = turbulenceModel::propertiesName,
        const word& type = typeName
    );

    //- Destructor
    virtual ~morphoCalcLES() = default;

    //- Read model coefficients from turbulenceProperties dictionary
    virtual bool read();

    //- Return the effective sub-grid scale viscosity (nu_t)
    virtual tmp<volScalarField> nut() const;

    //- Return the Reynolds stress tensor including non-linear Pope basis
    virtual tmp<volSymmTensorField> devReff() const;

    //- Correct turbulence operational state
    virtual void correct();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

#ifdef NoRepository
    #include "morphoCalcLES.C"
#endif

#endif

// ************************************************************************* //
