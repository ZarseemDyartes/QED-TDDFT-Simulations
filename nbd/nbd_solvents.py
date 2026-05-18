from pyscf import gto, scf, tdscf
import numpy as np
import time

# ==========================================
# Constants
# ==========================================
HARTREE_TO_EV = 27.211386245988
EV_NM = 1239.8419843320026

# ==========================================
# 1. MOLECULE SETUP
#    Geometry copied from nbd_base.py
# ==========================================
mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
C          -2.99554846        1.49184713        0.06984285
C          -3.08915556        0.07862534        0.10145860
C          -1.79815582        2.17809899        0.05826152
C          -0.58983022        1.41049177        0.07966505
C          -1.95291341       -0.72591027        0.12325400
C          -0.69349421       -0.02259677        0.11170980
N          -1.78904308        3.63135873        0.02520691
N          -1.95150461       -2.07986412        0.15360125
N           0.69806302        1.72629785        0.07703127
N           0.51633005       -0.54566576        0.12780684
O           1.35157792        0.52020757        0.10665545
O          -2.87356420        4.21273120        0.00856359
O          -0.69243507        4.18038272        0.01590057
H          -3.90624208        2.09109650        0.05328804
H          -4.07782349       -0.38361936        0.10860611
H          -2.81569835       -2.60270478        0.16240987
H          -1.07520465       -2.58348058        0.16777606
'''
mol.basis = '6-31+G**'
mol.charge = 0
mol.spin = 0
mol.build()

# ==========================================
# 2. ENVIRONMENT SETUP
#    PMMA-like ddCOSMO, matching the Coumarin solvent script style
# ==========================================
environments = [
    ("PMMA Matrix", 3.0),
    # Uncomment this if you also want a water comparison like the Coumarin script:
    # ("Water", 78.35),
]

# ==========================================
# Helper: print dominant TDA orbital transitions
# ==========================================
def dominant_tda_transitions(td, root_index, max_terms=5):
    """Return a compact string of the largest occupied -> virtual TDA amplitudes."""
    mo_occ = td._scf.mo_occ
    occ_indices = np.where(mo_occ > 0)[0]
    vir_indices = np.where(mo_occ == 0)[0]

    # For TDA, td.xy[root] is generally (X, Y), with Y absent/zero depending on PySCF version.
    xy_root = td.xy[root_index]
    X = xy_root[0] if isinstance(xy_root, (tuple, list)) else xy_root
    weights = np.abs(X) ** 2
    norm = np.sum(weights)
    if norm <= 0:
        return "unavailable"

    flat_order = np.argsort(weights.ravel())[::-1]
    terms = []
    for flat_idx in flat_order[:max_terms]:
        i_occ, a_vir = np.unravel_index(flat_idx, weights.shape)
        percent = 100.0 * weights[i_occ, a_vir] / norm
        # +1 converts zero-indexed Python MO numbering to normal chemistry-style numbering.
        occ_mo = occ_indices[i_occ] + 1
        vir_mo = vir_indices[a_vir] + 1
        terms.append(f"{occ_mo}->{vir_mo} ({percent:.1f}%)")
    return "; ".join(terms)

# ==========================================
# 3. LOOP THROUGH ENVIRONMENTS
# ==========================================
for env_name, eps_value in environments:
    start_time = time.time()

    print("\n" + "=" * 90)
    print(f"   STARTING NBD-AMINE CALCULATION: {env_name.upper()} (ddCOSMO eps = {eps_value})")
    print("=" * 90)

    # --- A. Ground State SCF with ddCOSMO ---
    print("\n--- Starting Ground State B3LYP/ddCOSMO SCF ---")
    mf = scf.RKS(mol)
    mf.xc = 'b3lyp'
    mf.max_cycle = 100
    mf.conv_tol = 1e-9

    # Activate ddCOSMO and set the static dielectric constant.
    mf = mf.ddCOSMO()
    mf.with_solvent.eps = eps_value
    mf.kernel()

    if not mf.converged:
        print("WARNING: SCF did not converge. TDDFT results may not be reliable.")

    # --- B. TD-DFT / TDA Calculation ---
    print(f"\n--- Running TDA-TDDFT for {env_name} ---")
    td = tdscf.TDA(mf)
    td.nroots = 10
    td.kernel()

    # --- C. Extract Transition Properties ---
    tdms = td.transition_dipole()

    print("\n" + "-" * 140)
    print(f" FINAL RESULTS: NBD-AMINE IN {env_name.upper()} ")
    print("-" * 140)
    print(
        f"{'State':<8} {'Energy (eV)':<13} {'Wavelength (nm)':<16} "
        f"{'mu_x':<11} {'mu_y':<11} {'mu_z':<11} {'|mu| (a.u.)':<13} {'f':<12} Dominant TDA transitions"
    )
    print("-" * 140)

    for i in range(td.nroots):
        tx, ty, tz = tdms[i]
        mu_mag = float(np.sqrt(tx**2 + ty**2 + tz**2))
        energy_ha = float(td.e[i])
        energy_ev = energy_ha * HARTREE_TO_EV
        wavelength_nm = EV_NM / energy_ev
        oscillator_strength = (2.0 / 3.0) * energy_ha * (mu_mag ** 2)
        dominant = dominant_tda_transitions(td, i, max_terms=3)

        print(
            f"S{i+1:<7} {energy_ev:<13.4f} {wavelength_nm:<16.2f} "
            f"{tx:<11.5f} {ty:<11.5f} {tz:<11.5f} {mu_mag:<13.5f} "
            f"{oscillator_strength:<12.6g} {dominant}"
        )

    elapsed = time.time() - start_time
    print("-" * 140)
    print(f"Elapsed wall time for {env_name}: {elapsed/60:.2f} minutes")
    print("-" * 140 + "\n")
