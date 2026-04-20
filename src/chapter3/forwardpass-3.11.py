# ==========================================
# Quantum Classifier (Class-Based Full Program)
# ==========================================

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ==========================================
# 1. Quantum Class Definition
# ==========================================

class QuantumClassifier:

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def circuit(self, x, theta):
        qc = QuantumCircuit(self.n_qubits)

        # Feature encoding
        for i in range(self.n_qubits):
            qc.ry(x[i], i)

        # Entanglement layer (assumes 2 qubits)
        qc.cx(0, 1)

        # Trainable layer
        for i in range(self.n_qubits):
            qc.ry(theta[i], i)

        return qc

    def forward(self, x, theta):
        qc = self.circuit(x, theta)
        state = Statevector.from_instruction(qc)

        # Z ⊗ I observable
        Z = np.kron([[1, 0], [0, -1]], np.eye(2))

        return np.real(state.expectation_value(Z))

    def predict_proba(self, x, theta):
        exp_val = self.forward(x, theta)
        return (1 - exp_val) / 2

    def predict(self, x, theta):
        prob = self.predict_proba(x, theta)
        return int(prob > 0.5)


# ==========================================
# 2. Dataset
# ==========================================

# Simple binary classification dataset
X = np.array([
    [0.1, 0.2],
    [0.2, 0.3],
    [1.0, 1.2],
    [1.1, 1.3]
])

Y = np.array([0, 0, 1, 1])


# ==========================================
# 3. Loss Function
# ==========================================

def loss(model, theta):
    preds = []

    for x in X:
        preds.append(model.predict_proba(x, theta))

    preds = np.array(preds)
    return np.mean((preds - Y) ** 2)


# ==========================================
# 4. Training Function
# ==========================================

def train(model, theta, lr=0.1, steps=100):
    for step in range(steps):
        grads = np.zeros_like(theta)
        eps = 1e-5

        # Finite-difference gradient
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_minus = theta.copy()

            theta_plus[i] += eps
            theta_minus[i] -= eps

            grad = (loss(model, theta_plus) - loss(model, theta_minus)) / (2 * eps)
            grads[i] = grad

        # Update parameters
        theta -= lr * grads

        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss(model, theta):.6f}")

    return theta


# ==========================================
# 5. Initialize Model
# ==========================================

model = QuantumClassifier(n_qubits=2)

theta = np.random.randn(2)

print("Initial Loss:", loss(model, theta))

# ==========================================
# 6. Train Model
# ==========================================

theta_trained = train(model, theta.copy())

print("\nTrained Parameters:", theta_trained)

# ==========================================
# 7. Evaluation
# ==========================================

print("\nPredictions:")
for x, y in zip(X, Y):
    prob = model.predict_proba(x, theta_trained)
    pred = model.predict(x, theta_trained)

    print(f"Input: {x}, True: {y}, Prob: {prob:.4f}, Pred: {pred}")

# ==========================================
# 8. Accuracy
# ==========================================

correct = 0
for x, y in zip(X, Y):
    if model.predict(x, theta_trained) == y:
        correct += 1

accuracy = correct / len(Y)
print("\nAccuracy:", accuracy)