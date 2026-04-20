from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt

def simple_model(x, theta):
    qc = QuantumCircuit(1)
    qc.ry(x, 0)
    qc.ry(theta, 0)
    state = Statevector.from_instruction(qc)
    Z = np.array([[1,0],[0,-1]])
    return np.real(state.expectation_value(Z))

# Plot decision boundary
xs = np.linspace(0, np.pi, 100)
ys = [simple_model(x, np.pi/4) for x in xs]

plt.plot(xs, ys)
plt.title("Quantum Decision Curve")
plt.show()
