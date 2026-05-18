import csv
import os
import time
import traceback

import numpy
from pyscf import gto, scf, tdscf
import qed

# ============================================================
# NBD-Amine QED omega x lambda grid scan
# Functional: B3LYP
# Solvent/environment: ddCOSMO epsilon = 3.0, PMMA-like matrix
# QED model: TDA-JC, matching the original nbd_lambdascan.py and
# nbd_omegascan.py setup.
#
# Frequency grid: original omega scan, 2.756 to 3.756 eV in 0.1 eV steps.
# Lambda grid: original lambda scan, 0.000 to 0.010 a.u. in 0.001 a.u. steps.
#
# Slurm-array behavior:
#   If SLURM_ARRAY_TASK_ID is set, each task runs one frequency and scans
#   all lambda values. Without SLURM_ARRAY_TASK_ID, the full grid is run.
# ============================================================

EV_CONVERSION = 27.2114
BASIS = "6-31+G**"
XC_FUNCTIONAL = "b3lyp"
N_PRE_ROOTS = 5
N_QED_ROOTS = 5
SOLVENT_EPSILON = 3.0
RESPONSE_MODEL = "TDA"
CAVITY_MODEL = "JC"

# Original lambda scan spacing: 0.000, 0.001, ..., 0.010 a.u.
LAMBDA_VALUES = numpy.linspace(0.0, 0.01, 11)

# Original omega scan frequencies: 2.756, 2.856, ..., 3.756 eV.
FREQUENCIES_EV = numpy.linspace(2.756, 3.756, 11)

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


def build_mean_field(mol):
    print(f"\n--- Starting Ground State SCF: {XC_FUNCTIONAL}/{BASIS}, ddCOSMO eps={SOLVENT_EPSILON} ---")
    mf = scf.RKS(mol).density_fit()
    mf.xc = XC_FUNCTIONAL
    mf = mf.ddCOSMO()
    mf.with_solvent.eps = SOLVENT_EPSILON
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mf


def get_prescan_direction(mf):
    print("\n--- Running standard TD-DFT/TDA pre-scan for TDM direction ---")
    td_std = tdscf.TDA(mf)
    td_std.nroots = N_PRE_ROOTS
    td_std.kernel()

    s1_energy_au = float(td_std.e[0])
    s1_energy_ev = s1_energy_au * EV_CONVERSION
    tdm_vector = td_std.transition_dipole()[0]
    tdm_magnitude = numpy.linalg.norm(tdm_vector)
    if tdm_magnitude < 1.0e-12:
        raise RuntimeError("S1 transition dipole is too small to define a cavity polarization direction.")
    tdm_direction = tdm_vector / tdm_magnitude

    print("\n>> AUTO-TUNING RESULTS <<")
    print(f"Computed S1 Energy : {s1_energy_au:.8f} a.u. ({s1_energy_ev:.4f} eV)")
    print(f"TDM Vector         : [{tdm_vector[0]:.6f}, {tdm_vector[1]:.6f}, {tdm_vector[2]:.6f}]")
    print(f"TDM Unit Direction : [{tdm_direction[0]:.6f}, {tdm_direction[1]:.6f}, {tdm_direction[2]:.6f}]")
    return s1_energy_au, s1_energy_ev, tdm_vector, tdm_direction


def selected_frequencies():
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    indexed_freqs = list(enumerate(FREQUENCIES_EV))
    if task_id is None:
        print("No SLURM_ARRAY_TASK_ID found: running all frequencies in one process.")
        return indexed_freqs

    task_id = int(task_id)
    if task_id < 0 or task_id >= len(indexed_freqs):
        raise ValueError(f"SLURM_ARRAY_TASK_ID={task_id} is outside valid range 0-{len(indexed_freqs)-1}.")
    print(f"SLURM_ARRAY_TASK_ID={task_id}: running one frequency and all lambda values.")
    return [indexed_freqs[task_id]]


def safe_energies_ev(td_obj):
    energies_au = numpy.asarray(td_obj.e)
    energies_ev = numpy.real_if_close(energies_au * EV_CONVERSION)
    return [float(numpy.real(x)) for x in energies_ev[:N_QED_ROOTS]]


def run_grid_scan():
    mol = build_mol()
    mf = build_mean_field(mol)
    s1_energy_au, s1_energy_ev, tdm_vector, tdm_direction = get_prescan_direction(mf)

    stamp = os.environ.get("SLURM_ARRAY_TASK_ID", "all")
    csv_name = f"nbd_omega_lambda_grid_b3lyp_pmma_{stamp}.csv"

    cavity_cls = getattr(qed, CAVITY_MODEL)
    response_cls = getattr(qed, RESPONSE_MODEL)

    with open(csv_name, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "functional", "basis", "solvent_model", "epsilon", "response_model", "cavity_model",
            "prescan_s1_ev", "frequency_index", "frequency_ev", "lambda_au", "root", "energy_ev",
            "status", "elapsed_seconds"
        ])

        for freq_idx, freq_ev in selected_frequencies():
            freq_au = float(freq_ev) / EV_CONVERSION
            cavity_freq = numpy.asarray([freq_au])

            print("\n" + "=" * 80)
            print(f"FREQUENCY INDEX {freq_idx}: omega={freq_ev:.4f} eV | MODEL={RESPONSE_MODEL}-{CAVITY_MODEL}")
            print("=" * 80)

            for lam in LAMBDA_VALUES:
                start = time.time()
                print(f"\n--- omega={freq_ev:.4f} eV | lambda={lam:.5f} a.u. ---")
                try:
                    cavity_mode = numpy.asarray([lam * tdm_direction])
                    cav_model = cavity_cls(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
                    td_qed = response_cls(mf, cav_obj=cav_model)
                    td_qed.nroots = N_QED_ROOTS
                    td_qed.kernel()

                    elapsed = time.time() - start
                    for root_idx, energy_ev in enumerate(safe_energies_ev(td_qed), start=1):
                        writer.writerow([
                            XC_FUNCTIONAL, BASIS, "ddCOSMO", SOLVENT_EPSILON, RESPONSE_MODEL, CAVITY_MODEL,
                            f"{s1_energy_ev:.10f}", freq_idx, f"{freq_ev:.10f}", f"{lam:.10f}",
                            root_idx, f"{energy_ev:.10f}", "ok", f"{elapsed:.2f}"
                        ])
                    handle.flush()
                except Exception as exc:
                    elapsed = time.time() - start
                    print("FAILED:", repr(exc))
                    traceback.print_exc()
                    writer.writerow([
                        XC_FUNCTIONAL, BASIS, "ddCOSMO", SOLVENT_EPSILON, RESPONSE_MODEL, CAVITY_MODEL,
                        f"{s1_energy_ev:.10f}", freq_idx, f"{freq_ev:.10f}", f"{lam:.10f}",
                        "", "", f"failed: {repr(exc)}", f"{elapsed:.2f}"
                    ])
                    handle.flush()

    print(f"\nDone. Wrote {csv_name}")


if __name__ == "__main__":
    run_grid_scan()
