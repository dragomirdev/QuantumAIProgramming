from qiskit.circuit.library import Initialize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt

def zz_feature_map(x):
    qc = QuantumCircuit(len(x))

    for i in range(len(x)):
        qc.ry(x[i], i)

    for i in range(len(x) - 1):
        qc.cx(i, i + 1)
        qc.rz(x[i] * x[i + 1], i + 1)
        qc.cx(i, i + 1)

    return qc

x = [np.pi/4, np.pi/2]   # length = 2 → valid (1 qubit)
qc = zz_feature_map(x)
state = Statevector.from_instruction(qc)
print(state.data)

qc.draw('mpl')
plt.show()