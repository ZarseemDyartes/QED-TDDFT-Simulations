import numpy
from pyscf import gto, scf, tdscf
import qed  # Your custom QED module

# ==========================================
# 1. MOLECULE SETUP: NBD-Amine
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
mol.build()

# ==========================================
# 2. GROUND STATE SCF (PMMA Matrix)
# ==========================================
print("\n--- Starting Ground State SCF ---")

# OPTIMIZATION 1: Density Fitting applied here
mf = scf.RKS(mol).density_fit() 
mf.xc = "b3lyp"
mf = mf.ddCOSMO()            
mf.with_solvent.eps = 3.0    
mf.kernel()

# ==========================================
# 3. PRE-SCAN: STANDARD TD-DFT
# ==========================================
print("\n--- Running Standard TD-DFT to find Resonance and Dipole ---")
td_std = tdscf.TDA(mf)

# OPTIMIZATION 2: Reduced roots
td_std.nroots = 5  
td_std.kernel()

# Extract the S1 Energy (in atomic units/Hartrees)
s1_energy_au = td_std.e[0]
s1_energy_ev = s1_energy_au * 27.2114

# Extract the S1 Transition Dipole Moment vector (X, Y, Z)
tdm_vector = td_std.transition_dipole()[0]

# Normalize the TDM vector to get a pure unit direction vector
tdm_magnitude = numpy.linalg.norm(tdm_vector)
tdm_direction = tdm_vector / tdm_magnitude

print(f"\n>> AUTO-TUNING RESULTS <<")
print(f"Resonant Cavity Freq : {s1_energy_au:.6f} a.u. ({s1_energy_ev:.3f} eV)")
print(f"TDM Vector (Raw)     : [{tdm_vector[0]:.4f}, {tdm_vector[1]:.4f}, {tdm_vector[2]:.4f}]")
print(f"TDM Direction (Unit) : [{tdm_direction[0]:.4f}, {tdm_direction[1]:.4f}, {tdm_direction[2]:.4f}]")

# Set the QED cavity frequency dynamically
cavity_freq = numpy.asarray([s1_energy_au])

# ==========================================
# 4. QED CAVITY LAMBDA SCAN
# ==========================================
print("\n--- Starting QED Lambda Scan ---")

# Scan over coupling strengths 0 to 0.01 in increments of 0.001
for coupling in numpy.linspace(0.0, 0.01, 11):
    print(f"\n========================================")
    print(f" RUNNING COUPLING STRENGTH: {coupling:.3f}")
    print(f"========================================")
    
    # Scale the pure direction vector by the current coupling strength
    scaled_cavity_mode = coupling * tdm_direction
    cavity_mode = numpy.asarray([scaled_cavity_mode])

    # TDA-JC Setup using dynamic frequency and mode
    cav_model = qed.JC(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
    td_qed    = qed.TDA(mf, cav_obj=cav_model)
    
    # OPTIMIZATION 3: Reduced roots in the loop
    td_qed.nroots = 5  
    
    # Run the QED TD-DFT calculation
    td_qed.kernel()