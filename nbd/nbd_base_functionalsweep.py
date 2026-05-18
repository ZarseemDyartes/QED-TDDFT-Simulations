import csv
import time
import traceback

from pyscf import gto, scf, tdscf

# ============================================================
# NBD-Amine normal TD-DFT/TDA functional sweep.
# This is NOT QED-TDDFT. It is intended to compare functionals
# for conjugated / nitro / possible charge-transfer excitations.
# ============================================================

EV_CONVERSION = 27.2114
BASIS = "6-31+G**"
NROOTS = 10
USE_DENSITY_FITTING = True
USE_DDCOSMO = False       # Set True if you want a PMMA-like matrix.
SOLVENT_EPSILON = 3.0

# Baseline + hybrids + range-separated/high-HF functionals.
# Unsupported aliases in your PySCF build are skipped cleanly.
FUNCTIONALS = [
    ("b3lyp", "baseline global hybrid; useful comparison to your original"),
    ("pbe0", "global hybrid; often robust for local valence excitations"),
    ("cam-b3lyp", "range-separated hybrid; better for charge-transfer/nitro systems"),
    ("wb97x", "range-separated hybrid; useful CT/conjugation comparison"),
    ("wb97x-d", "range-separated hybrid with empirical dispersion"),
    ("lc-wpbe", "long-range-corrected GGA hybrid"),
    ("m06-2x", "high-HF meta-hybrid; strong CT benchmark performer"),
    ("tpssh", "meta-hybrid comparison point with moderate exact exchange"),
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


def build_mol():
    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = ATOM_BLOCK
    mol.basis = BASIS
    mol.build()
    return mol


def build_mf(mol, xc_name):
    mf = scf.RKS(mol)
    if USE_DENSITY_FITTING:
        mf = mf.density_fit()
    mf.xc = xc_name
    if USE_DDCOSMO:
        mf = mf.ddCOSMO()
        mf.with_solvent.eps = SOLVENT_EPSILON
    return mf


def run_one_functional(mol, xc_name, notes, writer):
    print("\n" + "=" * 80)
    print(f"FUNCTIONAL: {xc_name} | {notes}")
    print("=" * 80)
    start = time.time()
    try:
        mf = build_mf(mol, xc_name)
        mf.kernel()

        td = tdscf.TDA(mf)
        td.nroots = NROOTS
        td.kernel()
        tdms = td.transition_dipole()
        elapsed = time.time() - start

        for idx in range(td.nroots):
            tx, ty, tz = tdms[idx]
            energy_ev = float(td.e[idx] * EV_CONVERSION)
            writer.writerow([
                xc_name, idx + 1, f"{energy_ev:.10f}",
                f"{tx:.10f}", f"{ty:.10f}", f"{tz:.10f}",
                "ok", f"{elapsed:.2f}", notes
            ])
        print(f"Completed {xc_name} in {elapsed:.1f} s")
    except Exception as exc:
        elapsed = time.time() - start
        print(f"FAILED {xc_name}: {repr(exc)}")
        traceback.print_exc()
        writer.writerow([xc_name, "", "", "", "", "", f"failed: {repr(exc)}", f"{elapsed:.2f}", notes])


def main():
    mol = build_mol()
    csv_name = "nbd_base_functional_sweep.csv"
    with open(csv_name, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "functional", "state", "energy_ev", "tdm_x", "tdm_y", "tdm_z",
            "status", "elapsed_seconds", "notes"
        ])
        for xc_name, notes in FUNCTIONALS:
            run_one_functional(mol, xc_name, notes, writer)
            handle.flush()
    print(f"\nDone. Wrote {csv_name}")


if __name__ == "__main__":
    main()
