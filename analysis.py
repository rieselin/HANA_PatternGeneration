# save as gray_scott_sweep.py and run with your ngsolve environment
import os
import itertools
import scipy as scipy
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from ngsolve import *
from netgen.occ import *
from netgen.occ import X, Y, Z as Xgen, Y, Z
from netgen.occ import OCCGeometry
from ngsolve.webgui import Draw
from datetime import datetime

# ---- User-tweakable sweep settings ----
dts = [ 10.0, 20.0, 40.0]              # list of dt values to test
n_doms = [50, 75]              # mesh resolution (n_dom in your code)
orders = [1, 2, 3]                     # polynomial orders to test
max_steps = 3000                    # ensure all runs record up to this many steps
tol = 1e-8                          # stopping tolerance (same as your code)
snapshot_eval_N = 300               # evaluation grid size for comparing solutions
output_dir = "sweep_output"
os.makedirs(output_dir, exist_ok=True)

# structured mesh umstellen

# ---- Fixed model parameters (from your script) ----
l_dom = 1.5
eps1 = Parameter(2e-5)
eps2 = Parameter(1e-5)
F = Parameter(0.056)
k = Parameter(0.065)

# ---- Helper: run one simulation with given parameters ----
def run_simulation(dt, n_dom, order, max_steps, tol, eval_points):
    """
    Returns dict with:
      'dt','n_dom','order',
      'deltanorms': list length <= max_steps,
      'final_u_grid', 'final_v_grid'  (both arrays shape eval_points.shape0),
      'stopping_step' (None if never below tol before max_steps),
      'steps_ran' (int)
    """
    # --- geometry and mesh ---
    rec = MoveTo(-l_dom/2, -l_dom/2).Rectangle(l_dom, l_dom).Face()
    rec.edges.Max(Xgen).name = 'right'
    rec.edges.Min(Xgen).name = 'left'
    rec.edges.Max(Y).name = 'top'
    rec.edges.Min(Y).name = 'bottom'
    right = rec.edges.Max(Xgen)
    rec.edges.Min(Xgen).Identify(right, name="left")
    top = rec.edges.Max(Y)
    rec.edges.Min(Y).Identify(top, name="bottom")
    geo = OCCGeometry(rec, dim=2)
    mesh = Mesh(geo.GenerateMesh(maxh=l_dom/n_dom))

    for i in range(3):
        mesh.Refine()
    # print summary
    print(f"Sim: dt={dt}, n_dom={n_dom}, order={order} -> ne={mesh.ne}, nv={mesh.nv}")

    # --- FE spaces ---
    V = Periodic(H1(mesh, order=order))
    X = V * V
    u, v = X.TrialFunction()
    w, q = X.TestFunction()

    gfx = GridFunction(X)
    gfu, gfv = gfx.components
    gfxold = GridFunction(X)
    gfuold, gfvold = gfxold.components

    # initial conditions (copy your logic)
    l_init = l_dom / n_dom * 20
    gfuold.Set(IfPos((l_init/2)**2 - x*x,
                     IfPos((l_init/2)**2 - y*y, 0.5, 1), 1))
    gfvold.Set(IfPos((l_init/2)**2 - x*x,
                     IfPos((l_init/2)**2 - y*y, 0.25, 0), 0))
    gfxold.vec[:] += 0.01 * np.random.normal(size=X.ndof)
    gfx.vec.data = gfxold.vec


    a = BilinearForm(X, symmetric=True)
    a += eps1 * grad(u) * grad(w) * dx
    a += eps2 * grad(v) * grad(q) * dx
    a.Assemble()

    m = BilinearForm(X, symmetric=True)
    m += u*w*dx
    m += v*q*dx
    m.Assemble()

    # build M* = M + dt*A (matrix)
    mstar = a.mat.CreateMatrix()
    mstar.AsVector().data = m.mat.AsVector() + dt * a.mat.AsVector()
    mstarinv = mstar.Inverse(inverse="sparsecholesky")

    # linear form f (explicit nonlinearity)
    f = LinearForm(X)
    f += dt*(-gfuold*gfvold**2 + F*(1-gfuold))*w*dx(bonus_intorder=4)
    f += dt*(gfuold*gfvold**2 - (k+F)*gfvold)*q*dx(bonus_intorder=4)

    # time integration
    deltanorms = []
    res = gfx.vec.CreateVector()
    deltauv = gfx.vec.CreateVector()

    scene_u = None  # no Draw to speed batch runs

    stopping_step = None
    steps_ran = 0

    with TaskManager():
        for j in range(max_steps):
            f.Assemble()
            res.data = -dt * a.mat * gfxold.vec + f.vec
            deltauv.data = mstarinv * res
            gfx.vec.data = gfxold.vec + deltauv
            gfxold.vec.data = gfx.vec

            deltan = deltauv.Norm()
            deltanorms.append(deltan)
            steps_ran += 1

            if deltan < tol:
                stopping_step = j
                break

    # Evaluate final u,v on eval_points (a list/array of (x,y) coords)
    Xi, Yi = eval_points
    pts = mesh(Xi.flatten(), Yi.flatten())
    u_final = gfu(pts).reshape(Xi.shape)
    v_final = gfv(pts).reshape(Xi.shape)

    return {
        'dt': dt,
        'n_dom': n_dom,
        'order': order,
        'deltanorms': np.array(deltanorms),
        'final_u_grid': u_final,
        'final_v_grid': v_final,
        'stopping_step': stopping_step,
        'steps_ran': steps_ran
    }

# ---- Prepare evaluation grid (common grid for all comparisons) ----
Npix = snapshot_eval_N
xi = np.linspace(-l_dom/2, l_dom/2, Npix)
Xi, Yi = np.meshgrid(xi, xi)

# ---- Run sweep ----
combos = list(itertools.product(dts, n_doms, orders))
results = []
print("Running sweep of", len(combos), "simulations...")

for dt_val, n_dom_val, order_val in combos:
    res = run_simulation(dt_val, n_dom_val, order_val, max_steps, tol, (Xi, Yi))
    results.append(res)

# ---- Find first common step where ALL runs have Δ < tol ----
# consistent arrays up to max_steps; pad with last observed value (or +inf if not run)
all_deltas = []
for r in results:
    arr = r['deltanorms']
    if len(arr) < max_steps:
        # pad with the last value repeated so indexing is safe (or a small number if they stopped early)
        pad_val = arr[-1] if len(arr) > 0 else np.inf
        arr_padded = np.concatenate([arr, np.full(max_steps - len(arr), pad_val)])
    else:
        arr_padded = arr[:max_steps]
    all_deltas.append(arr_padded)
all_deltas = np.vstack(all_deltas)  # shape (n_runs, max_steps)

# at step j, find max across runs
max_across = np.max(all_deltas, axis=0)
common_stop = None
for j in range(max_steps):
    if max_across[j] < tol:
        common_stop = j
        break

print("Common stopping step (first j where max across runs < tol):", common_stop)

# ---- Pairwise comparisons of final u fields ----
n_runs = len(results)
final_u_vecs = [r['final_u_grid'].flatten() for r in results]
final_v_vecs = [r['final_v_grid'].flatten() for r in results]

# metrics: L2, L-inf, IoU on thresholded u
def l2_norm(a, b):
    return np.sqrt(np.mean((a - b)**2))
def linf_norm(a, b):
    return np.max(np.abs(a - b))
def iou_binary(a, b, thr=0.5):
    A = a > thr
    B = b > thr
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return inter / union if union > 0 else 1.0

pairwise_metrics = []
for i in range(n_runs):
    for j in range(i+1, n_runs):
        u_i = final_u_vecs[i]; u_j = final_u_vecs[j]
        l2 = l2_norm(u_i, u_j)
        linf = linf_norm(u_i, u_j)
        iou = iou_binary(u_i, u_j, thr=0.5)
        pairwise_metrics.append({
            'i': i, 'j': j,
            'params_i': (results[i]['dt'], results[i]['n_dom'], results[i]['order']),
            'params_j': (results[j]['dt'], results[j]['n_dom'], results[j]['order']),
            'L2': l2, 'Linf': linf, 'IoU': iou
        })

# ---- Save a summary table and plots ----
import json
summary = {
    'run_summary': [
        {
            'index': idx,
            'dt': r['dt'],
            'n_dom': r['n_dom'],
            'order': r['order'],
            'stopping_step': r['stopping_step'],
            'steps_ran': r['steps_ran']
        } for idx, r in enumerate(results)
    ],
    'common_stop': common_stop,
    'pairwise_metrics': pairwise_metrics
}
with open(os.path.join(output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), 'w') as f:
    json.dump(summary, f, indent=2)

# plot max Δ across runs and per-run Δs for diagnostics
plt.figure(figsize=(8,4))
plt.semilogy(max_across, label='max Δ across runs')
plt.axhline(tol, color='k', linestyle='--', label=f'tol={tol}')
plt.xlabel('time step index j')
plt.ylabel('Δ (deltauvNorm)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'convergence_max_across_runs.png'))
plt.close()

# save example final fields (first few runs)
for idx, r in enumerate(results[:min(6, len(results))]):
    plt.figure(figsize=(4,4))
    plt.imshow(r['final_u_grid'] > 0.5, origin='lower', extent=[-l_dom/2, l_dom/2, -l_dom/2, l_dom/2])
    plt.title(f"u > 0.5: dt={r['dt']}, n={r['n_dom']}, ord={r['order']}")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'final_u_binary_run{idx}.png'))
    plt.close()

print("Sweep finished. Results and plots saved in", output_dir)
