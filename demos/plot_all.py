import os
import numpy as np
import matplotlib.pyplot as plt

# Folder where all outputs are
base_dir = r"C:\Users\timur\Desktop\TU Wien\WS25\NSSC1\project_3\ASC-ODE\build"

# Map filenames -> labels
files = {
    "output_test_ode_nystrom.txt":                   "Nystrom (explicit)",
    "output_test_ode_rk2.txt":               "RK2 (explicit)",
    "output_test_ode_rk4.txt":               "RK4 (explicit)",
    #"output_test_ode_gauss.txt":             "Gauss (implicit, fixed c)",
    #"output_test_ode_gauss_legendre.txt":    "Gauss-Legendre (implicit)",
    #"output_test_ode_gauss_radau.txt":       "Radau (implicit)",
}

# Load all that exist
solutions = {}  # label -> data array (t, x, v)
for fname, label in files.items():
    path = os.path.join(base_dir, fname)
    if not os.path.isfile(path):
        print(f"Skipping {fname} (not found)")
        continue

    data = np.loadtxt(path, usecols=(0, 1, 2))
    solutions[label] = data
    print(f"Loaded {fname} as '{label}' with shape {data.shape}")

# ----------------------------------------------------------------------
# 1) Plot position and velocity vs time for all methods
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 5))

for label, data in solutions.items():
    t = data[:, 0]
    x = data[:, 1]
    v = data[:, 2]

    plt.plot(t, x, label=f"{label} – position")
    plt.plot(t, v, linestyle='--', label=f"{label} – velocity")

plt.xlabel("time")
plt.ylabel("value")
plt.title("Mass–spring: position and velocity vs time (all methods)")
plt.legend()
plt.grid(True)

# ----------------------------------------------------------------------
# 2) Phase plots for all methods (x vs v)
# ----------------------------------------------------------------------
plt.figure(figsize=(6, 6))
for label, data in solutions.items():
    x = data[:, 1]
    v = data[:, 2]
    plt.plot(x, v, label=label)

plt.xlabel("position")
plt.ylabel("velocity")
plt.title("Mass-Spring: phase plot (all methods)")
plt.legend()
plt.grid(True)
plt.axis("equal")

# ----------------------------------------------------------------------
# 3) Error vs exact solution (difference between methods and truth)
# ----------------------------------------------------------------------
# Exact solution for unit mass, unit stiffness, IC x(0)=1, v(0)=0:
# x(t) = cos(t), v(t) = -sin(t)
plt.figure(figsize=(8, 5))
for label, data in solutions.items():
    t = data[:, 0]
    x_num = data[:, 1]
    v_num = data[:, 2]
    x_exact = np.cos(t)
    v_exact = -np.sin(t)
    err = np.sqrt((x_num - x_exact)**2 + (v_num - v_exact)**2)
    plt.plot(t, err, label=label)

plt.xlabel("time")
plt.ylabel("L2 error in (x,v)")
plt.title("Error vs exact solution (all methods)")
plt.yscale("log")   # error usually looks better on log scale
plt.legend()
plt.grid(True)


plt.show()
