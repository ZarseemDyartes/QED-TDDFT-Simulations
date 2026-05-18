import sys
from pyscf import gto, scf
# Using geomeTRIC, which is the modern standard optimizer for PySCF
from pyscf.geomopt.geometric_solver import optimize

# ==========================================
# 1. MOLECULE SETUP
# ==========================================
print("\n--- Loading NBD Geometry ---")
mol = gto.Mole()
mol.verbose = 4

# PySCF can read the coordinates directly from your external XYZ file
mol.atom = 'nbd_init.xyz'
mol.basis = 'cc-pVDZ'
mol.build()

# ==========================================
# 2. GROUND STATE SCF SETUP
# ==========================================
print("\n--- Setting up Ground State SCF ---")
mf = scf.RKS(mol).density_fit() 
mf.xc = 'b3lyp'

# ==========================================
# 3. GEOMETRY OPTIMIZATION
# ==========================================
print("\n--- Starting Geometry Optimization ---")
# The optimizer will repeatedly solve the SCF, calculate gradients, 
# and move the atoms until they reach the lowest energy resting state.
mol_eq = optimize(mf)

# ==========================================
# 4. SAVE FINAL GEOMETRY
# ==========================================
print("\n--- Optimization Complete ---")
# Write the newly optimized coordinates out to a new file so you can use them!
mol_eq.tofile('nbd_opt.xyz')
print("Optimized coordinates successfully saved to 'nbd_opt.xyz'")