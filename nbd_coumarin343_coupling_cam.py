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

c343_geom_base = '''
C  -5.585634936212   2.449611799944  -0.108879421850
C  -6.783460232473   1.586846081547  -0.515400637148
C  -4.295306387782   1.650195044050  -0.034596819912
C  -3.077403775857   2.317183411925   0.042735644801
C  -1.850471879212   1.640828521538   0.118599860258
C  -1.885276849692   0.229925444899   0.111609356109
C  -4.301265651884   0.225391168137  -0.007587868323
C  -3.074870898092  -0.493695488375   0.063321383699
C  -6.746298487741   0.251557680893   0.212727114100
N  -5.508830459103  -0.467567187524  -0.073042510076
C  -5.528480365905  -1.877300709293   0.301621979384
C  -4.360364524784  -2.629461557838  -0.322180994132
C  -3.028912808607  -2.006941737028   0.097386940415
C  -0.569775809633   2.290102704164   0.269113407765
C   0.586007291879   1.590318188058   0.171078504617
C   0.473916344833   0.128710416213  -0.209792459188
O  -0.722046270754  -0.486461880561   0.260182635294
C   1.884577981730   2.301081742180   0.305750034224
O   1.987499794631   3.491732063241   0.493258110712
O   3.019016769256   1.548482530893   0.190269460062
H  -5.780679370150   2.924105424196   0.870145478067
H  -5.460796101992   3.275724099954  -0.828107446884
H  -6.764746928343   1.401071262084  -1.602501243836
H  -7.730099663336   2.100509577133  -0.285765836484
H  -3.072119697350   3.411319787037   0.046433034007
H  -7.582692886718  -0.385707935957  -0.114502080531
H  -6.871049780199   0.408397357128   1.307170776071
H  -6.480254127235  -2.297290952574  -0.060014051477
H  -5.524872528369  -1.999237274547   1.407333011361
H  -4.466488540970  -2.601859593502  -1.420123665609
H  -4.401970754761  -3.685791564498  -0.014625596849
H  -2.213526307804  -2.361423254384  -0.550916933848
H  -2.760131388788  -2.341661613720   1.116514988635
H  -0.521913958251   3.362458487980   0.472863665495
H   1.274113029748  -0.480261905448   0.246490484170
H   0.555198862707   0.000417370408  -1.311885932202
H   2.794749546669   0.610277343832   0.101979889355
'''

# Translate Coumarin 343 by 20 Angstroms along the Z-axis
c343_geom_shifted = ""
for line in c343_geom_base.strip().split('\n'):
    parts = line.split()
    atom = parts[0]
    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
    c343_geom_shifted += f"{atom} {x:.10f} {y:.10f} {z + 20.0:.10f}\n"

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
print(f"NBD S1 Frequency: {nbd_freq_au * 27.2114:.4f} eV ({nbd_freq_au:.6f} a.u.)")

# ==========================================
# 3. MONOMER 2: COUMARIN 343 ALONE
# ==========================================
print("\n--- Running TD-DFT on Isolated C343 ---")
mol_c343 = gto.Mole(atom=c343_geom_shifted, basis=basis_set, verbose=3).build()
mf_c343 = scf.RKS(mol_c343).density_fit()
mf_c343.xc = "b3lyp"
mf_c343.kernel()

td_c343 = tdscf.TDA(mf_c343)
td_c343.nroots = 1
td_c343.kernel()
c343_freq_au = td_c343.e[0]
print(f"C343 S1 Frequency: {c343_freq_au * 27.2114:.4f} eV ({c343_freq_au:.6f} a.u.)")

# ==========================================
# 4. DIMER SETUP & SCF
# ==========================================
print("\n--- Running Ground State SCF for Dimer (20 A Separation) ---")
dimer_geom = nbd_geom.strip() + "\n" + c343_geom_shifted.strip()
mol_dimer = gto.Mole(atom=dimer_geom, basis=basis_set, verbose=3).build()

mf_dimer = scf.RKS(mol_dimer).density_fit()
mf_dimer.xc = "cam-b3lyp"
mf_dimer.kernel()

# Standard TD-DFT on dimer to align cavity mode
td_std = tdscf.TDA(mf_dimer)
td_std.nroots = 3
td_std.kernel()

tdm_vector = td_std.transition_dipole()[0]
tdm_magnitude = np.linalg.norm(tdm_vector)
tdm_direction = tdm_vector / tdm_magnitude 

# ==========================================
# 5. QED CAVITY Test
# ==========================================
print("\n--- Starting QED Cavity Test ---")

coupling_strength = 0.001
cavity_mode_vec = coupling_strength * tdm_direction
cavity_mode = np.asarray([cavity_mode_vec])

midpoint_freq_au = (nbd_freq_au + c343_freq_au) / 2.0
scan_freqs_au =  [midpoint_freq_au]
labels = ["Midpoint Resonance"]

print(f"Locked Cavity Coupling: {coupling_strength} a.u.")

for label, freq_au in zip(labels, scan_freqsau):
    freq_ev = freq_au * 27.2114
    print(f"\n========================================")
    print(f" TARGET: {label}")
    print(f" RUNNING CAVITY FREQUENCY: {freq_ev:.4f} eV")
    print(f"========================================")
    
    cavity_freq = np.asarray([freq_au])

    # TDA-JC Setup
    cav_model = qed.JC(mf_dimer, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
    td_qed    = qed.TDA(mf_dimer, cav_obj=cav_model)
    
    # Requesting 5 roots to track the lowest states
    td_qed.nroots = 5  
    td_qed.kernel()