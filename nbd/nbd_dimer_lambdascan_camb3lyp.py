import numpy
from pyscf import gto, scf, tdscf
import qed

# ==========================================
# 1. MOLECULE SETUP: NBD-Amine
# ==========================================
print("\n--- Building the C343 Supermolecule ---")

c343_single = '''
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

# Automatically generate the second molecule shifted by 20 Å on the Z-axis
c343_dimer = c343_single
for line in c343_single.strip().split('\n'):
    parts = line.split()
    atom = parts[0]
    x = float(parts[1])
    y = float(parts[2])
    z = float(parts[3]) + 20.0
    c343_dimer += f"\n{atom}  {x:.10f}  {y:.10f}  {z:.10f}"

mol = gto.Mole()
mol.verbose = 4
mol.atom = c343_dimer
mol.basis = '6-31+G**'
mol.build()

# ==========================================
# 2. GROUND STATE SCF
# ==========================================
print("\n--- Starting Ground State SCF (Dimer) ---")
# Using density fitting to speed up the double-sized calculation
mf = scf.RKS(mol).density_fit() 
mf.xc = "cam-b3lyp"  # range-separated hybrid functional
mf = mf.ddCOSMO()            
mf.with_solvent.eps = 3.0    
mf.kernel()

# ==========================================
# 3. PRE-SCAN: STANDARD TD-DFT (For Alignment)
# ==========================================
print("\n--- Running Standard TD-DFT to find Collective Resonance ---")
td_std = tdscf.TDA(mf)
# Requesting more roots because the density of states is higher for a dimer
td_std.nroots = 10  
td_std.kernel()

# Grab the energy of the first bright collective state
s1_energy_au = td_std.e[0]
cavity_freq = numpy.asarray([s1_energy_au])

# Extract the collective Transition Dipole Moment vector
tdm_vector = td_std.transition_dipole()[0]
tdm_magnitude = numpy.linalg.norm(tdm_vector)
tdm_direction = tdm_vector / tdm_magnitude

print(f"\n>> ALIGNMENT RESULTS <<")
print(f"Resonant Cavity Freq : {s1_energy_au:.6f} a.u.")
print(f"TDM Direction (Unit) : [{tdm_direction[0]:.4f}, {tdm_direction[1]:.4f}, {tdm_direction[2]:.4f}]")

# ==========================================
# 4. QED LAMBDA SCAN
# ==========================================
print("\n--- Starting QED Lambda Scan (Collective Coupling) ---")

# Scan lambda from 0.00 to 0.02 a.u.
lambda_values = numpy.linspace(0.00, 0.02, 11)

for lam in lambda_values:
    print(f"\n========================================")
    print(f" RUNNING LAMBDA COUPLING: {lam:.4f} a.u.")
    print(f"========================================")
    
    # Scale the cavity mode vector by the current lambda
    scaled_cavity_mode = lam * tdm_direction
    cavity_mode = numpy.asarray([scaled_cavity_mode])

    # TDA-JC Setup using scanned lambda and locked frequency
    cav_model = qed.JC(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
    td_qed    = qed.TDA(mf, cav_obj=cav_model)
    td_qed.nroots = 10  
    
    # Run the QED TD-DFT calculation
    td_qed.kernel()