import csv
import time
import traceback

import numpy
from pyscf import gto, scf, tdscf

# ============================================================
# NBD-Amine base TD-DFT/TDA basis-set sweep
# Functional: B3LYP
# Response: normal PySCF TDA, not QED-TDDFT
#
# This is modeled after nbd_base.py, but loops over a compact
# set of basis sets useful for checking excited-state sensitivity.
# ============================================================

EV_CONVERSION = 27.2114
XC_FUNCTIONAL = "b3lyp"
NROOTS = 10
USE_DENSITY_FITTING = True
USE_DDCOSMO = False  # keep this False to match the original nbd_base.py vacuum-style calculation
SOLVENT_EPSILON = 3.0

BASIS_SETS = [
    ("6-31Gstar", "6-31G*"),
    ("6-31plusGstarstar", "6-31+G**"),
    ("def2-SVP", "def2-SVP"),
    ("def2-TZVP", "def2-TZVP"),
]

ATOM_BLOCK = """
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
"""


def build_mol(basis_name):
    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = ATOM_BLOCK
    mol.basis = basis_name
    mol.build()
    return mol


def build_mean_field(mol):
    mf = scf.RKS(mol)
    if USE_DENSITY_FITTING:
        mf = mf.density_fit()
    mf.xc = XC_FUNCTIONAL
    if USE_DDCOSMO:
        mf = mf.ddCOSMO()
        mf.with_solvent.eps = SOLVENT_EPSILON
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mf


def run_tda(mf):
    td = tdscf.TDA(mf)
    td.nroots = NROOTS
    td.kernel()
    return td


def run_basis_sweep():
    csv_name = "nbd_base_basis_sweep_b3lyp.csv"
    with open(csv_name, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "basis_label", "basis", "functional", "use_ddcosmo", "epsilon",
            "state", "energy_ev", "tdm_x", "tdm_y", "tdm_z", "tdm_norm",
            "status", "elapsed_seconds"
        ])

        for basis_label, basis_name in BASIS_SETS:
            start = time.time()
            print("\n" + "=" * 80)
            print(f"RUNNING BASIS: {basis_label} ({basis_name}) | XC={XC_FUNCTIONAL}")
            print("=" * 80)
            try:
                mol = build_mol(basis_name)
                mf = build_mean_field(mol)
                td = run_tda(mf)
                tdms = td.transition_dipole()
                elapsed = time.time() - start

                print("\n" + "-" * 72)
                print(f"{'State':<8} {'Energy (eV)':<14} {'X':<12} {'Y':<12} {'Z':<12} {'|TDM|':<12}")
                print("-" * 72)
                for i in range(min(NROOTS, len(td.e))):
                    tx, ty, tz = tdms[i]
                    tdm_norm = float(numpy.linalg.norm(tdms[i]))
                    energy_ev = float(td.e[i] * EV_CONVERSION)
                    print(f"{i+1:<8} {energy_ev:<14.6f} {tx:<12.6f} {ty:<12.6f} {tz:<12.6f} {tdm_norm:<12.6f}")
                    writer.writerow([
                        basis_label, basis_name, XC_FUNCTIONAL, USE_DDCOSMO, SOLVENT_EPSILON,
                        i + 1, f"{energy_ev:.10f}", f"{tx:.10f}", f"{ty:.10f}", f"{tz:.10f}",
                        f"{tdm_norm:.10f}", "ok", f"{elapsed:.2f}"
                    ])
                handle.flush()
            except Exception as exc:
                elapsed = time.time() - start
                print("FAILED:", repr(exc))
                traceback.print_exc()
                writer.writerow([
                    basis_label, basis_name, XC_FUNCTIONAL, USE_DDCOSMO, SOLVENT_EPSILON,
                    "", "", "", "", "", "", f"failed: {repr(exc)}", f"{elapsed:.2f}"
                ])
                handle.flush()

    print(f"\nDone. Wrote {csv_name}")


if __name__ == "__main__":
    run_basis_sweep()
