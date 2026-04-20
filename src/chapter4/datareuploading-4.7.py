from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt

def reuploading_model(x, theta):
    qc = QuantumCircuit(1)

    for i in range(len(theta)):
        qc.ry(x[i % len(x)], 0)
        qc.ry(theta[i], 0)

    return qc

x = [np.pi/4, np.pi/2]
theta = [0.1, 0.2, 0.3]

qc = reuploading_model(x, theta)

state = Statevector.from_instruction(qc)
print(state.data)

qc.draw('mpl')
plt.show()