import scipy as scipy
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import pandas as pd

from ngsolve.meshes import MakeStructured2DMesh
from ngsolve import *

# ---------------------------------------
# Configuration
# ---------------------------------------
dts = [5.0] #, 5.0, 10.0 => test without varying dt
n_doms = [5, 10, 20] # could be replaced with refine order 
orders = [1, 2, 3]

# Fixed parameters
l_dom = 1.5
eps1_val = 2e-5
eps2_val = 1e-5
F_val = 0.05
k_val = 0.065

# Simulation parameters
T_final = 2000.0  # Final time for all simulations
max_newton_iter = 10
newton_tol = 1e-8

# Reference solution parameters (finest grid)
dt_ref = 5.0
n_dom_ref = 40
order_ref = 4

# ---------------------------------------
# Helper Functions
# ---------------------------------------

def create_mesh(l_dom, n_dom):
    """Create structured 2D mesh with quadrilaterals or triangles"""
    mesh = MakeStructured2DMesh(
        quads=False, nx=n_dom, ny=n_dom,
        mapping=lambda x, y: (x * l_dom - l_dom/2, y * l_dom - l_dom/2)
    )
    return mesh

def set_initial_conditions(gfuold, gfvold, l_dom, n_dom):
    """Set initial conditions with small perturbation"""
    l_init = l_dom / n_dom * 20

    gfuold.Set(IfPos((l_init/2)**2 - x*x,
                     IfPos((l_init/2)**2 - y*y, 0.5, 1), 1))
    gfvold.Set(IfPos((l_init/2)**2 - x*x,
                     IfPos((l_init/2)**2 - y*y, 0.25, 0), 0))

    # Add same random seed for reproducibility
    np.random.seed(42)

    ndof = gfuold.space.ndof + gfvold.space.ndof
    return 0.01 * np.random.normal(size=ndof)

def run_simulation(dt, n_dom, order, T_final, verbose=False):
    """Run Gray-Scott simulation with given parameters"""
    
    # Create mesh
    mesh = create_mesh(l_dom, n_dom)

    # Setup finite element space
    V = Periodic(H1(mesh, order=order))
    X = V * V
    u, v = X.TrialFunction()
    w, q = X.TestFunction()
    
    # Grid functions
    gfx = GridFunction(X)
    gfu, gfv = gfx.components
    gfxold = GridFunction(X)
    gfuold, gfvold = gfxold.components
    
    # Initial conditions
    perturbation = set_initial_conditions(gfuold, gfvold, l_dom, n_dom)
    gfxold.vec[:] += perturbation
    gfx.vec.data = gfxold.vec
    
    # Parameters
    eps1 = Parameter(eps1_val)
    eps2 = Parameter(eps2_val)
    F = Parameter(F_val)
    k = Parameter(k_val)
    
    # Assemble matrices
    a = BilinearForm(X, symmetric=True)
    a += eps1 * grad(u) * grad(w) * dx
    a += eps2 * grad(v) * grad(q) * dx
    a.Assemble()
    
    m = BilinearForm(X, symmetric=True)
    m += u*w*dx
    m += v*q*dx
    m.Assemble()
    
    # M* = M + dt/2*A (for trapezoidal rule)
    mstar = m.mat.CreateMatrix()
    mstar.AsVector().data = m.mat.AsVector() + dt * a.mat.AsVector()
    mstarinv = mstar.Inverse(inverse="sparsecholesky")

    f = LinearForm(X)
    f += dt*(-gfuold*gfvold**2+F*(1-gfuold))*w*dx(bonus_intorder=4)
    f += dt*(gfuold*gfvold**2-(k+F)*gfvold)*q*dx(bonus_intorder=4)

    res = gfx.vec.CreateVector()
    deltaufv = GridFunction(X)   # increment as GridFunction
    deltauf, deltavf = deltaufv.components
    
    # Time stepping
    nsteps = int(T_final / dt)
    t = 0.0
    
    delta_norms = []
    time_points = []
    
    with TaskManager():
        for j in range(nsteps):
            t = j * dt  # update time
            
            # Assemble right-hand side f (explicit nonlinearity)
            f.Assemble()

            # Solve M* δ = -dt*A*u_old + f
            res.data = -dt * a.mat * gfxold.vec + f.vec

            delta_vec = mstarinv * res
            deltaufv.vec.data = delta_vec
            gfx.vec.data = gfxold.vec + deltaufv.vec

            try:
                # If the integrator accepts expressions with gridfunction components:
                deltau_norm_sq = Integrate(deltauf**2 + deltavf**2, mesh, bonus_intorder=4)
                deltauvNorm = np.sqrt(deltau_norm_sq)
            except Exception:
                # Fallback: measure coefficient Euclidean norm (less precise but robust)
                deltauvNorm = float(deltaufv.vec.Norm())

            # Update for next step
            gfxold.vec.data = gfx.vec

            # Store norm history
            delta_norms.append(deltauvNorm)
            time_points.append(t)

            if verbose and j % 100 == 0:
                print(f"  Step {j}/{nsteps}, t={t:.2f}, ||delta||={deltauvNorm:.6e}", end='\r')
            
            # Optional early stopping if stationary
            # if deltauvNorm < newton_tol:
            #     if verbose:
            #         print(f"\n  Reached stationary solution at t={t:.2f}")
            #     break
    
    if verbose:
        print(f"\nSimulation completed: dt={dt}, n_dom={n_dom}, order={order}")
    
    # Return sampled values instead of GridFunctions for error computation
    Npixel = 200
    xi = np.linspace(-l_dom/2, l_dom/2, Npixel)
    Xi, Yi = np.meshgrid(xi, xi)
    mips = mesh(Xi.flatten(), Yi.flatten())
    
    u_vals = gfu(mips).reshape(Npixel, Npixel)
    v_vals = gfv(mips).reshape(Npixel, Npixel)
    
    return u_vals, v_vals, np.array(delta_norms), np.array(time_points)


def compute_L2_error(u1_vals, u2_vals):
    """Compute L2 error between two sampled solutions"""
    diff = u1_vals - u2_vals
    l2_error = np.sqrt(np.sum(diff**2))
    return l2_error

# ---------------------------------------
# Main Error Analysis
# ---------------------------------------


# Compute reference solution
print("\n[1/2] Computing reference solution...")
print(f"    Parameters: dt={dt_ref}, n_dom={n_dom_ref}, order={order_ref}")
u_ref, v_ref, delta_norms_ref, time_points_ref = run_simulation(
    dt_ref, n_dom_ref, order_ref, T_final, verbose=True
)

# Plot reference delta norm evolution
fig_ref, ax_ref = plt.subplots(figsize=(10, 6))
ax_ref.semilogy(time_points_ref, delta_norms_ref)
ax_ref.set_xlabel('Time')
ax_ref.set_ylabel('||delta|| (norm of increment)')
ax_ref.set_title('Convergence to Stationary Solution (Reference)')
ax_ref.grid(True)
plt.tight_layout()

output_path = "output/"
os.makedirs(output_path, exist_ok=True)
date_time_tag = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
plt.savefig(f'{output_path}stationary_convergence_{date_time_tag}.pdf')
print(f"Stationary convergence plot saved")

# Store results
results = []

print("\n[2/2] Running parameter study...")
total_runs = len(dts) * len(n_doms) * len(orders)
run_count = 0

for dt in dts:
    for n_dom in n_doms:
        for order in orders:
            run_count += 1
            print(f"\n--- Run {run_count}/{total_runs} ---")
            print(f"    dt={dt}, n_dom={n_dom}, order={order}")

            # Run simulation
            u_vals, v_vals, delta_norms, time_points = run_simulation(
                dt, n_dom, order, T_final, verbose=True
            )

            # Compute errors against reference
            l2_err_u = compute_L2_error(u_vals, u_ref)
            l2_err_v = compute_L2_error(v_vals, v_ref)

            print(f"    L2 error (u): {l2_err_u:.6e}")
            print(f"    L2 error (v): {l2_err_v:.6e}")

            results.append({
                'dt': dt,
                'n_dom': n_dom,
                'order': order,
                'h': l_dom/n_dom,
                'ndof': (n_dom+1)**2,  # Approximate for structured mesh
                'l2_err_u': l2_err_u,
                'l2_err_v': l2_err_v,
                'final_delta_norm': delta_norms[-1]
            })

# ---------------------------------------
# Save and visualize results
# ---------------------------------------

df = pd.DataFrame(results)

# Save results
csv_file = f'{output_path}error_analysis_{date_time_tag}.csv'
df.to_csv(csv_file, index=False)
print(f"\n\nResults saved to: {csv_file}")

# Create convergence plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Error vs dt (fixed n_dom, order)
ax = axes[0, 0]
for n_dom in n_doms:
    for order in orders:
        subset = df[(df['n_dom'] == n_dom) & (df['order'] == order)]
        if len(subset) > 0:
            ax.loglog(subset['dt'], subset['l2_err_u'], 'o-', 
                     label=f'n={n_dom}, p={order}')
ax.set_xlabel('dt')
ax.set_ylabel('L2 Error (u)')
ax.set_title('Temporal Convergence (u)')
ax.legend(fontsize=8)
ax.grid(True)

# Plot 2: Error vs h (fixed dt, varying order)
ax = axes[0, 1]
for dt in dts:
    for order in orders:
        subset = df[(df['dt'] == dt) & (df['order'] == order)]
        if len(subset) > 0:
            ax.loglog(subset['h'], subset['l2_err_u'], 'o-', 
                     label=f'dt={dt}, p={order}')
ax.set_xlabel('h (mesh size)')
ax.set_ylabel('L2 Error (u)')
ax.set_title('Spatial Convergence (u)')
ax.legend(fontsize=8)
ax.grid(True)

# Plot 3: Error vs order (fixed dt, n_dom)
ax = axes[0, 2]
for dt in dts:
    for n_dom in n_doms:
        subset = df[(df['dt'] == dt) & (df['n_dom'] == n_dom)]
        if len(subset) > 0:
            ax.semilogy(subset['order'], subset['l2_err_u'], 'o-', 
                       label=f'dt={dt}, n={n_dom}')
ax.set_xlabel('Polynomial Order')
ax.set_ylabel('L2 Error (u)')
ax.set_title('p-Convergence (u)')
ax.legend(fontsize=8)
ax.grid(True)

# Plots 4-6: Same for v component
ax = axes[1, 0]
for n_dom in n_doms:
    for order in orders:
        subset = df[(df['n_dom'] == n_dom) & (df['order'] == order)]
        if len(subset) > 0:
            ax.loglog(subset['dt'], subset['l2_err_v'], 'o-', 
                     label=f'n={n_dom}, p={order}')
ax.set_xlabel('dt')
ax.set_ylabel('L2 Error (v)')
ax.set_title('Temporal Convergence (v)')
ax.legend(fontsize=8)
ax.grid(True)

ax = axes[1, 1]
for dt in dts:
    for order in orders:
        subset = df[(df['dt'] == dt) & (df['order'] == order)]
        if len(subset) > 0:
            ax.loglog(subset['h'], subset['l2_err_v'], 'o-', 
                     label=f'dt={dt}, p={order}')
ax.set_xlabel('h (mesh size)')
ax.set_ylabel('L2 Error (v)')
ax.set_title('Spatial Convergence (v)')
ax.legend(fontsize=8)
ax.grid(True)

ax = axes[1, 2]
for dt in dts:
    for n_dom in n_doms:
        subset = df[(df['dt'] == dt) & (df['n_dom'] == n_dom)]
        if len(subset) > 0:
            ax.semilogy(subset['order'], subset['l2_err_v'], 'o-', 
                       label=f'dt={dt}, n={n_dom}')
ax.set_xlabel('Polynomial Order')
ax.set_ylabel('L2 Error (v)')
ax.set_title('p-Convergence (v)')
ax.legend(fontsize=8)
ax.grid(True)

plt.tight_layout()
plot_file = f'{output_path}convergence_plots_{date_time_tag}.pdf'
plt.savefig(plot_file)
print(f"Convergence plots saved to: {plot_file}")
plt.show()

# Print summary table
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(df.to_string(index=False))
print("\nAnalysis complete!")