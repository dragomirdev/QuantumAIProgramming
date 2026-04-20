from qiskit.circuit.library import Initialize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt

def amplitude_encoding(x):
    norm = np.linalg.norm(x)
    x = x / norm
    n_qubits = int(np.log2(len(x)))
    qc = QuantumCircuit(n_qubits)
    qc.append(Initialize(x), qc.qubits)
    return qc

x = [np.pi/4, np.pi/2]   # length = 2 → valid (1 qubit)
qc = amplitude_encoding(x)
state = Statevector.from_instruction(qc)
print(state.data)

qc.draw('mpl')
plt.show()