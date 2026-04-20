import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit

qc = QuantumCircuit(1)

theta = np.pi / 4
qc.ry(theta, 0)

state = Statevector.from_instruction(qc)
print(state.data)
