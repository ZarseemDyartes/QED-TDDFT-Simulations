import numpy
from pyscf import gto, scf, tdscf
import qed  # Your custom QED module

# ==========================================
# 1. MOLECULE SETUP: Coumarin 343
# ==========================================
mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
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