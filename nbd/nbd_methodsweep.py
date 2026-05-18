import csv
import os
import time
import traceback

import numpy
from pyscf import gto, scf, tdscf
import qed

# ============================================================
# NBD-Amine QED scan: B3LYP, named cavity frequencies, all 8
# QED-TDDFT/TDA variants exposed by cc-ats/qed-tddft.
#
# Slurm-array behavior:
#   If SLURM_ARRAY_TASK_ID is set, this script runs exactly one
#   (model, named_frequency) pair and scans all lambda values.
#   Without SLURM_ARRAY_TASK_ID, it runs the full cartesian product.
# ============================================================

EV_CONVERSION = 27.2114
BASIS = "6-31+G**"
XC_FUNCTIONAL = "b3lyp"
N_PRE_ROOTS = 5
N_QED_ROOTS = 5
USE_DDCOSMO = True
SOLVENT_EPSILON = 3.0  # approximate PMMA dielectric constant

# Lambda values match your original nbd_lambdascan.py: 0.000 to 0.010 a.u.
LAMBDA_VALUES = numpy.linspace(0.0, 0.01, 11)

# These are the 11 detuning points from your omega scan, now named.
# The additional "auto_s1" entry uses the S1 energy computed in the pre-scan.
NAMED_FREQUENCIES_EV = [
    ("auto_s1", None),
    ("detune_m0p50", 2.756),
    ("detune_m0p40", 2.856),
    ("detune_m0p30", 2.956),
    ("detune_m0p20", 3.056),
    ("detune_m0p10", 3.156),
    ("fixed_3p256", 3.256),
    ("detune_p0p10", 3.356),
    ("detune_p0p20", 3.456),
    ("detune_p0p30", 3.556),
    ("detune_p0p40", 3.656),
    ("detune_p0p50", 3.756),
]

# Four cavity Hamiltonian variants x two response levels = eight models.
MODEL_SPECS = [
    ("TDA_PF", "TDA", "PF"),
    ("TDA_Rabi", "TDA", "Rabi"),
    ("TDA_RWA", "TDA", "RWA"),
    ("TDA_JC", "TDA", "JC"),
    ("TDDFT_PF", "TDDFT", "PF"),
    ("TDDFT_Rabi", "TDDFT", "Rabi"),
    ("TDDFT_RWA", "TDDFT", "RWA"),
    ("TDDFT_JC", "TDDFT", "JC"),
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


def build_mean_field(mol):
    print(f"\n--- Starting Ground State SCF: {XC_FUNCTIONAL}/{BASIS} ---")
    mf = scf.RKS(mol).density_fit()
    mf.xc = XC_FUNCTIONAL
    if USE_DDCOSMO:
        mf = mf.ddCOSMO()
        mf.with_solvent.eps = SOLVENT_EPSILON
    mf.kernel()
    return mf


def get_prescan(mf):
    print("\n--- Running standard TD-DFT/TDA pre-scan for S1 and TDM direction ---")
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
    print(f"S1 Energy: {s1_energy_au:.8f} a.u. ({s1_energy_ev:.4f} eV)")
    print(f"TDM Vector: [{tdm_vector[0]:.6f}, {tdm_vector[1]:.6f}, {tdm_vector[2]:.6f}]")
    print(f"TDM Unit Direction: [{tdm_direction[0]:.6f}, {tdm_direction[1]:.6f}, {tdm_direction[2]:.6f}]")
    return s1_energy_au, s1_energy_ev, tdm_vector, tdm_direction


def make_cavity_model(mf, cavity_name, cavity_mode, cavity_freq):
    cavity_cls = getattr(qed, cavity_name)
    return cavity_cls(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)


def make_response_model(mf, response_name, cav_model):
    response_cls = getattr(qed, response_name)
    return response_cls(mf, cav_obj=cav_model)


def select_work_items():
    items = [(model_idx, freq_idx)
             for model_idx in range(len(MODEL_SPECS))
             for freq_idx in range(len(NAMED_FREQUENCIES_EV))]

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        print("No SLURM_ARRAY_TASK_ID found: running ALL model/frequency combinations in one process.")
        return items

    task_id = int(task_id)
    if task_id < 0 or task_id >= len(items):
        raise ValueError(f"SLURM_ARRAY_TASK_ID={task_id} is outside valid range 0-{len(items)-1}.")

    print(f"SLURM_ARRAY_TASK_ID={task_id}: running one model/frequency combination.")
    return [items[task_id]]


def safe_energies_ev(td_obj):
    energies_au = numpy.asarray(td_obj.e)
    energies_ev = numpy.real_if_close(energies_au * EV_CONVERSION)
    return [float(numpy.real(x)) for x in energies_ev[:N_QED_ROOTS]]


def run_scan():
    mol = build_mol()
    mf = build_mean_field(mol)
    s1_energy_au, s1_energy_ev, tdm_vector, tdm_direction = get_prescan(mf)

    work_items = select_work_items()
    stamp = os.environ.get("SLURM_ARRAY_TASK_ID", "all")
    csv_name = f"nbd_lambdascan_freqs_all8_b3lyp_{stamp}.csv"

    with open(csv_name, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "response", "cavity_model", "frequency_name", "frequency_ev",
            "lambda_au", "root", "energy_ev", "status", "elapsed_seconds"
        ])

        for model_idx, freq_idx in work_items:
            model_label, response_name, cavity_name = MODEL_SPECS[model_idx]
            freq_name, freq_ev = NAMED_FREQUENCIES_EV[freq_idx]
            if freq_ev is None:
                freq_ev = s1_energy_ev
            freq_au = freq_ev / EV_CONVERSION
            cavity_freq = numpy.asarray([freq_au])

            print("\n" + "=" * 80)
            print(f"MODEL: {model_label} | FREQ: {freq_name} = {freq_ev:.4f} eV")
            print("=" * 80)

            for lam in LAMBDA_VALUES:
                start = time.time()
                print(f"\n--- {model_label} | {freq_name} | lambda={lam:.5f} a.u. ---")
                try:
                    cavity_mode = numpy.asarray([lam * tdm_direction])
                    cav_model = make_cavity_model(mf, cavity_name, cavity_mode, cavity_freq)
                    td_qed = make_response_model(mf, response_name, cav_model)
                    td_qed.nroots = N_QED_ROOTS
                    td_qed.kernel()

                    energies_ev = safe_energies_ev(td_qed)
                    elapsed = time.time() - start
                    for root_idx, energy_ev in enumerate(energies_ev, start=1):
                        writer.writerow([
                            model_label, response_name, cavity_name, freq_name, f"{freq_ev:.8f}",
                            f"{lam:.8f}", root_idx, f"{energy_ev:.10f}", "ok", f"{elapsed:.2f}"
                        ])
                    handle.flush()
                except Exception as exc:
                    elapsed = time.time() - start
                    print("FAILED:", repr(exc))
                    traceback.print_exc()
                    writer.writerow([
                        model_label, response_name, cavity_name, freq_name, f"{freq_ev:.8f}",
                        f"{lam:.8f}", "", "", f"failed: {repr(exc)}", f"{elapsed:.2f}"
                    ])
                    handle.flush()

    print(f"\nDone. Wrote {csv_name}")


if __name__ == "__main__":
    run_scan()
