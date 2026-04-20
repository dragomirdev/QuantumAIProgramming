from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)

state = Statevector.from_instruction(qc)
print(state.data)

qc.draw('mpl')
plt.show()