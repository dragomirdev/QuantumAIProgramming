import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit

def quantum_model(x, theta):
    qc = QuantumCircuit(1)
    qc.ry(x, 0)
    qc.ry(theta, 0)
    return qc

def expectation(qc):
    state = Statevector.from_instruction(qc)
    Z = np.array([[1,0],[0,-1]])
    return np.real(state.expectation_value(Z))

qc = QuantumCircuit(1)
print (expectation(qc))