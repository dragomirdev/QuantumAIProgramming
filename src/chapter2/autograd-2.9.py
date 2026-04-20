import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def circuit(x):
    qml.RY(x, wires=0)
    return qml.expval(qml.PauliZ(0))

theta = np.array(0.5, requires_grad=True)

def cost(t):
    return circuit(t)

grad = qml.grad(cost)
print(grad(theta))
