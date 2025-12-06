import numpy as np
import matplotlib.pyplot as plt

# Load CSV file
data = np.genfromtxt("results/legendre_output.csv", delimiter=",", names=True)

x = data["x"]

plt.figure(figsize=(10, 6))

# Plot polynomials P0...P5
for i in range(6):
    P = data[f"P{i}"]
    plt.plot(x, P, label=f"P{i}")

plt.title("Legendre Polynomials P0..P5")
plt.xlabel("x")
plt.ylabel("P_n(x)")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(6):
    dP = data[f"dP{i}"]
    plt.plot(x, dP, label=f"P{i}'")
plt.title("Derivatives of Legendre Polynomials")
plt.xlabel("x")
plt.ylabel("dP_n(x)/dx")
plt.legend()
plt.grid(True)
plt.show()