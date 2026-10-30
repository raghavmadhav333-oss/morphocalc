# morphoCalcLES: Production OpenFOAM Drop-In Large Eddy Simulation Plugin

**Author:** MorphoCalc-Nexus Research Initiative (Mukund)  
**Turbulence Closure:** Non-Linear S.B. Pope 10-Tensor Integrity Basis with Second Law Positivity  
**Target Architecture:** OpenFOAM-v2012, v2112, v2212, v2312, OpenFOAM-9/10/11

---

## 1. Overview

`morphoCalcLES` is a production C++ dynamic shared library plugin for **OpenFOAM**. It replaces classical linear eddy-viscosity models (Smagorinsky, WALE, dynamic k-eqn) with the autonomously discovered **S.B. Pope 3D non-linear frame-indifferent tensor integrity closure**:

$$\tau_{\text{SGS}} = -2\nu_t S_{ij} + G_2(I_1, I_2) T_{ij}^{(2)} + G_3 T_{ij}^{(3)}$$

where:
- $T_{ij}^{(1)} = S_{ij}$ (Linear traceless rate of strain)
- $T_{ij}^{(2)} = S_{ik}\Omega_{kj} - \Omega_{ik}S_{kj}$ (Vorticity-strain non-linear cross coupling)
- $T_{ij}^{(3)} = S_{ik}S_{kj} - \frac{1}{3}\text{tr}(S^2)\delta_{ij}$ (Quadratic normal stress generator)
- $G_2(I_1, I_2) = -0.0095 + 0.0156 I_1 - 0.0142 |I_2|$
- $G_3 = -10.3033$

---

## 2. Compilation Instructions

In any standard OpenFOAM terminal:

```bash
cd morphocalc/openfoam
wmake libso
```

This compiles the source code into:
`$FOAM_USER_LIBBIN/libmorphoCalcLES.so`

---

## 3. Usage in Any OpenFOAM Case

In your case's `system/controlDict`, load the dynamic library:

```cpp
libs
(
    "libmorphoCalcLES.so"
);
```

In `constant/turbulenceProperties`:

```cpp
simulationType LES;

LES
{
    LESModel        morphoCalcLES;
    turbulence      on;
    printCoeffs     on;

    morphoCalcLESCoeffs
    {
        Ck          0.094;
        Ce          1.048;
        alpha_g2   -0.0095;
        beta_g2     0.0156;
        gamma_g2   -0.0142;
        c_g3      -10.3033;
    }
}
```

Then run your standard solver (`pimpleFoam`, `pisoFoam`, or `rhoPimpleFoam`):

```bash
pimpleFoam
```

Zero neural network runtime overhead. Pure compiled C++ tensor algebra.
