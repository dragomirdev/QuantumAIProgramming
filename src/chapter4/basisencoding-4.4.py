from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt

def basis_encoding(x):
    qc = QuantumCircuit(len(x))
    for i, val in enumerate(x):
        if val == 1:
            qc.x(i)
    return qc

x = [np.pi/4]
qc = basis_encoding(x)
state = Statevector.from_instruction(qc)
print(state.data)

qc.draw('mpl')
plt.show()