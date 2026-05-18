import numpy
from pyscf import gto, scf, tdscf
import qed

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
# 2. GROUND STATE SCF
# ==========================================
print("\n--- Starting Ground State SCF ---")
mf = scf.RKS(mol).density_fit() 
mf.xc = "b3lyp"
mf = mf.ddCOSMO()            
mf.with_solvent.eps = 3.0    
mf.kernel()

# ==========================================
# 3. PRE-SCAN: Find Resonance & Dipole Direction
# ==========================================
print("\n--- Running Standard TD-DFT to find Resonance and Dipole ---")
td_std = tdscf.TDA(mf)
td_std.nroots = 5  
td_std.kernel()

s1_energy_au = td_std.e[0]
tdm_vector = td_std.transition_dipole()[0]
tdm_magnitude = numpy.linalg.norm(tdm_vector)
tdm_direction = tdm_vector / tdm_magnitude

print(f"\n>> ALIGNMENT RESULTS <<")
print(f"Resonant Cavity Freq : {s1_energy_au:.6f} a.u.")
print(f"TDM Direction (Unit) : [{tdm_direction[0]:.4f}, {tdm_direction[1]:.4f}, {tdm_direction[2]:.4f}]")

cavity_freq = numpy.asarray([s1_energy_au])

# ==========================================
# 4. QED POLARIZATION ROTATION SCAN
# ==========================================
print("\n--- Starting QED Polarization Rotation Scan ---")

fixed_coupling = 0.01

# Scan from 0 degrees to 90 degrees in increments of 10
angles_deg = numpy.arange(0, 91, 10)

for angle in angles_deg:
    print(f"\n========================================")
    print(f" RUNNING MISALIGNMENT ANGLE: {angle}°")
    print(f"========================================")
    
    # Convert degrees to radians for the math
    theta = numpy.radians(angle)
    
    # Create a rotation matrix around the Z-axis
    c, s = numpy.cos(theta), numpy.sin(theta)
    rotation_matrix = numpy.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
    # Rotate the unit vector by theta
    rotated_direction = rotation_matrix.dot(tdm_direction)
    
    # Apply the fixed coupling strength to the newly rotated vector
    scaled_cavity_mode = fixed_coupling * rotated_direction
    cavity_mode = numpy.asarray([scaled_cavity_mode])

    print(f"Rotated Cavity Mode Vector : [{cavity_mode[0][0]:.5f}, {cavity_mode[0][1]:.5f}, {cavity_mode[0][2]:.5f}]")

    # TDA-JC Setup 
    cav_model = qed.JC(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
    td_qed    = qed.TDA(mf, cav_obj=cav_model)
    td_qed.nroots = 5  
    
    # Run QED TD-DFT
    td_qed.kernel()