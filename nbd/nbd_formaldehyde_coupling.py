import numpy as np
from pyscf import gto, scf, tdscf
import qed

# ==========================================
# 1. MOLECULE DEFINITIONS & TRANSLATION
# ==========================================
nbd_geom = '''
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

# Your specified Formaldehyde geometry
h2co_geom_base = '''
H       -0.9450370725    -0.0000000000     1.1283908757
C       -0.0000000000     0.0000000000     0.5267587663
H        0.9450370725     0.0000000000     1.1283908757
O        0.0000000000    -0.0000000000    -0.6771667936
'''

# Translate Formaldehyde 20 Angstroms along the Z-axis
h2co_geom_shifted = ""
for line in h2co_geom_base.strip().split('\n'):
    parts = line.split()
    atom = parts[0]
    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
    h2co_geom_shifted += f"{atom} {x:.10f} {y:.10f} {z + 20.0:.10f}\n"

basis_set = '6-31+G**'

# ==========================================
# 2. MONOMER 1: NBD ALONE
# ==========================================
print("\n--- Running TD-DFT on Isolated NBD ---")
mol_nbd = gto.Mole(atom=nbd_geom, basis=basis_set, verbose=3).build()
mf_nbd = scf.RKS(mol_nbd).density_fit()
mf_nbd.xc = "b3lyp"
mf_nbd.kernel()

td_nbd = tdscf.TDA(mf_nbd)
td_nbd.nroots = 1
td_nbd.kernel()
nbd_freq_au = td_nbd.e[0]
nbd_freq_ev = nbd_freq_au * 27.2114
print(f"NBD S1 Frequency: {nbd_freq_ev:.4f} eV ({nbd_freq_au:.6f} a.u.)")

# ==========================================
# 3. MONOMER 2: FORMALDEHYDE ALONE
# ==========================================
print("\n--- Running TD-DFT on Isolated Formaldehyde ---")
mol_h2co = gto.Mole(atom=h2co_geom_shifted, basis=basis_set, verbose=3).build()
mf_h2co = scf.RKS(mol_h2co).density_fit()
mf_h2co.xc = "b3lyp"
mf_h2co.kernel()

td_h2co = tdscf.TDA(mf_h2co)
td_h2co.nroots = 1
td_h2co.kernel()
h2co_freq_au = td_h2co.e[0]
h2co_freq_ev = h2co_freq_au * 27.2114
print(f"Formaldehyde S1 Frequency: {h2co_freq_ev:.4f} eV ({h2co_freq_au:.6f} a.u.)")

# ==========================================
# 4. DIMER SETUP & SCF
# ==========================================
print("\n--- Running Ground State SCF for Dimer (20 A Separation) ---")
dimer_geom = nbd_geom.strip() + "\n" + h2co_geom_shifted.strip()
mol_dimer = gto.Mole(atom=dimer_geom, basis=basis_set, verbose=3).build()

mf_dimer = scf.RKS(mol_dimer).density_fit()
mf_dimer.xc = "b3lyp"
mf_dimer.kernel()

# Standard TD-DFT on dimer to align cavity mode
td_std = tdscf.TDA(mf_dimer)
td_std.nroots = 3
td_std.kernel()

tdm_vector = td_std.transition_dipole()[0]
tdm_magnitude = np.linalg.norm(tdm_vector)
tdm_direction = tdm_vector / tdm_magnitude 

# ==========================================
# 5. QED CAVITY FREQUENCY SCAN
# ==========================================
print("\n--- Starting QED Cavity Frequency Scan ---")

coupling_strength = 0.01
cavity_mode_vec = coupling_strength * tdm_direction
cavity_mode = np.asarray([cavity_mode_vec])

print(f"Locked Cavity Coupling: {coupling_strength} a.u.")
print(f"Scanning 5 points between {nbd_freq_ev:.4f} eV and {h2co_freq_ev:.4f} eV")

# Create 5 frequency points between the two monomer states
start_freq = min(nbd_freq_au, h2co_freq_au)
end_freq = max(nbd_freq_au, h2co_freq_au)
scan_freqs_au = np.linspace(start_freq, end_freq, 5)

for freq_au in scan_freqs_au:
    freq_ev = freq_au * 27.2114
    print(f"\n========================================")
    print(f" RUNNING CAVITY FREQUENCY: {freq_ev:.4f} eV")
    print(f"========================================")
    
    cavity_freq = np.asarray([freq_au])

    # TDA-JC Setup
    cav_model = qed.JC(mf_dimer, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
    td_qed    = qed.TDA(mf_dimer, cav_obj=cav_model)
    
    # Requesting 5 roots to ensure we capture both states in the dimer
    td_qed.nroots = 5  
    td_qed.kernel()