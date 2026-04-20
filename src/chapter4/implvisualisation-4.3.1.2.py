from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt

def angle_encoding(x):
    qc = QuantumCircuit(len(x))
    for i, val in enumerate(x):
        qc.ry(val, i)
    return qc

x = [np.pi/4]
qc = angle_encoding(x)
state = Statevector.from_instruction(qc)
print(state.data)

qc.draw('mpl')
plt.show()
