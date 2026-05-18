# [Coumarin 343](https://www.sigmaaldrich.com/US/en/product/aldrich/393029): 
# Preliminary Tests 
1. **Base**: B3LYP/cc-pVDZ optimized geometry, B3LYP/6-31+G** single-point KS-DFT + TDA(TDDFT) without solvent effects. 
2. **Solvents (TDA)**: B3LYP/cc-pVDZ optimized geometry, B3LYP/6-31+G** single-point KS-DFT + TDA(TDDFT) with ddCOSMO to simulate solvent effects from PMMA and water.
# Basic QED-TDDFT Tests
For all these tests, assume B3LYP/cc-pVDZ optimized geometry, B3LYP/6-31+G** single-point KS-DFT (density fitting with aug-cc-pVDZ) and TDA(TDDFT) for preliminary calculations, and TDA-JC for QED-TDDFT calculations. 

Standard cavity factors: 
- Cavity Strength - 0.01 a.u.
- Cavity Frequency - Tuned to first excitation resonance, as calculated using TDA(TDDFT) with ddCOSMO for PMMA.
- Cavity Polarization - Tuned to transition dipole moment, as calculated using TDA(TDDFT) with ddCOSMO for PMMA.

3. **Lambda Scan**: Scan over 11 different cavity strengths (lambdas) from 0 to 0.01 a.u (spaced by 0.001 a.u.)
4. **Omega Scan**: Scan over 11 different cavity frequencies (omegas) centered on the first excitation resonance (spaced by 0.1 eV)
5. **Polarization Scan**: Scan over 10 different rotation angles of the cavity polarization vector about the z-axis from 0 to 90 degrees (spaced by 10 degrees)
