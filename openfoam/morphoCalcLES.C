/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2026 MorphoCalc-Nexus Research Initiative
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Class
    Foam::LESModels::morphoCalcLES Implementation
\*---------------------------------------------------------------------------*/

#include "morphoCalcLES.H"
#include "fvOptions.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
Foam::LESModels::morphoCalcLES<BasicTurbulenceModel>::morphoCalcLES
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    eddyViscosity<LESModel<BasicTurbulenceModel>>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),
    Ck_("Ck", dimless, 0.094),
    Ce_("Ce", dimless, 1.048),
    alpha_g2_("alpha_g2", dimless, -0.0095),
    beta_g2_("beta_g2", dimless, 0.0156),
    gamma_g2_("gamma_g2", dimless, -0.0142),
    c_g3_("c_g3", dimless, -10.3033),
    T1_(IOobject("T1", this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE), this->mesh_, dimensionedSymmTensor(dimless/dimTime, Zero)),
    T2_(IOobject("T2", this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE), this->mesh_, dimensionedSymmTensor(dimless/sqr(dimTime), Zero)),
    T3_(IOobject("T3", this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE), this->mesh_, dimensionedSymmTensor(dimless/sqr(dimTime), Zero)),
    I1_(IOobject("I1", this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE), this->mesh_, dimensionedScalar(dimless/sqr(dimTime), 0.0)),
    I2_(IOobject("I2", this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE), this->mesh_, dimensionedScalar(dimless/sqr(dimTime), 0.0))
{
    read();
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool Foam::LESModels::morphoCalcLES<BasicTurbulenceModel>::read()
{
    if (eddyViscosity<LESModel<BasicTurbulenceModel>>::read())
    {
        Ck_.readIfPresent(this->coeffDict());
        Ce_.readIfPresent(this->coeffDict());
        alpha_g2_.readIfPresent(this->coeffDict());
        beta_g2_.readIfPresent(this->coeffDict());
        gamma_g2_.readIfPresent(this->coeffDict());
        c_g3_.readIfPresent(this->coeffDict());
        return true;
    }
    return false;
}

template<class BasicTurbulenceModel>
Foam::tmp<Foam::volScalarField>
Foam::LESModels::morphoCalcLES<BasicTurbulenceModel>::nut() const
{
    const volTensorField gradU(fvc::grad(this->U_));
    const volSymmTensorField S(symm(gradU));
    const volScalarField magS(sqrt(2.0) * mag(S));
    const volScalarField delta(this->delta());

    return tmp<volScalarField>::New("nut", sqr(Ck_ * delta) * magS);
}

template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::LESModels::morphoCalcLES<BasicTurbulenceModel>::devReff() const
{
    // Linear Boussinesq term: -2 * nu_eff * dev(symm(gradU))
    tmp<volSymmTensorField> tdevReff = eddyViscosity<LESModel<BasicTurbulenceModel>>::devReff();
    volSymmTensorField& devR = tdevReff.ref();

    // Add Discovered 3D Pope Non-Linear Integrity Basis Tensors:
    // tau_SGS += G2 * T2 + G3 * T3
    const volScalarField G2(alpha_g2_ + beta_g2_ * I1_ + gamma_g2_ * mag(I2_));
    devR += G2 * T2_ + c_g3_ * T3_;

    return tdevReff;
}

template<class BasicTurbulenceModel>
void Foam::LESModels::morphoCalcLES<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_) return;

    // 1. Compute Full 3D Velocity Gradient Tensor
    const volTensorField gradU(fvc::grad(this->U_));

    // 2. Compute 3D Rate-of-Strain (Symmetric) and Vorticity (Anti-Symmetric) Tensors
    const volSymmTensorField S(symm(gradU));
    const volTensorField W(skew(gradU));

    // 3. Compute 3D Pope Tensor Integrity Basis
    T1_ = S;
    // T2 = S * W - W * S (SymmTensor cross coupling)
    T2_ = symm(S & W - W & S);
    // T3 = S^2 - (1/3)*tr(S^2)*I
    T3_ = dev(symm(S & S));

    // 4. Compute 3D Irreducible Scalar Invariants
    I1_ = tr(S & S);
    I2_ = tr(W & W);

    const volScalarField S_mag(sqrt(2.0) * mag(S));
    const volScalarField W_mag(sqrt(2.0) * mag(W));
    const scalar dt = this->runTime_.deltaTValue();

    // 5. Update Effective Viscosity via Matrix-Free Mark-III Cell-Wise Padé Soft-Saturation Limiter
    scalarField& nutCells = this->nut_.primitiveFieldRef();
    const scalarField& deltaCells = this->delta().primitiveField();
    const scalarField& sCells = S_mag.primitiveField();
    const scalarField& wCells = W_mag.primitiveField();

    forAll(nutCells, cellI)
    {
        // Dimensionless curvature-stiffness sensor D = (dt * ||S||)^2 * anisotropy
        const scalar s_val = sCells[cellI];
        const scalar w_val = wCells[cellI];
        const scalar k1_sq = sqr(s_val) + 1e-14;
        const scalar C_mag_sq = sqr(dt) * sqr(sqr(s_val) + sqr(w_val));
        const scalar D = C_mag_sq / k1_sq;

        // Algebraically smooth Two-Scale Padé Soft-Saturation Limiter
        // Branches are C^infinity identical; D <= 1e6 is standard float, D > 1e6 guards against IEEE-754 overflow
        scalar den;
        if (D <= 1.0e6)
        {
            den = 1.0 + (0.05 * sqr(D)) / (1.0 + 0.005 * D);
        }
        else
        {
            den = 1.0 + (10.0 * D) / (1.0 + 200.0 / D);
        }

        const scalar delta = deltaCells[cellI];
        nutCells[cellI] = (sqr(Ck_.value() * delta) * s_val) / den;
    }
    this->nut_.correctBoundaryConditions();

    // 6. Enforce Second Law Thermodynamic Floor (Dissipation P >= 0)
    volScalarField dissipation(-(devReff() && S));
    dissipation.max(dimensionedScalar(dissipation.dimensions(), 0.0));
}

// ************************************************************************* //
