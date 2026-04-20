from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt

def entangled_feature_map(x):
    qc = QuantumCircuit(len(x))

    for i, val in enumerate(x):
        qc.ry(val, i)

    for i in range(len(x) - 1):
        qc.cx(i, i + 1)

    return qc

x = [np.pi/4, np.pi/2]

qc = entangled_feature_map(x)
state = Statevector.from_instruction(qc)

print(state.data)

qc.draw('mpl')
plt.show()