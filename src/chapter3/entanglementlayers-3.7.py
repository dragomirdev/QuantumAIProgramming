# ==============================
# Full Quantum AI Program
# ==============================

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ==============================
# 1. Quantum Model Definition
# ==============================

def two_qubit_model(x, theta):
    qc = QuantumCircuit(2)

    # Feature encoding
    qc.ry(x[0], 0)
    qc.ry(x[1], 1)

    # Entanglement layer
    qc.cx(0, 1)

    # Trainable layer
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)

    return qc


# ==============================
# 2. Expectation Value Function
# ==============================

def expectation_z0(qc):
    state = Statevector.from_instruction(qc)

    # Z ⊗ I operator
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    Z0 = np.kron(Z, I)

    return np.real(state.expectation_value(Z0))


# ==============================
# 3. Prediction Function
# ==============================

def predict(x, theta):
    qc = two_qubit_model(x, theta)
    exp_val = expectation_z0(qc)

    # Map expectation value to probability
    prob = (1 - exp_val) / 2
    return prob


# ==============================
# 4. Example Input
# ==============================

x = np.array([0.5, 1.0])  # Input features
theta = np.array([0.1, 0.2])  # Trainable parameters

print("Prediction:", predict(x, theta))


# ==============================
# 5. Simple Loss Function
# ==============================

def loss(x, theta, y_true):
    y_pred = predict(x, theta)
    return (y_pred - y_true) ** 2


# ==============================
# 6. Simple Training Loop (Numerical Gradient)
# ==============================

def train(x, y_true, theta, lr=0.1, steps=50):
    for step in range(steps):
        grads = np.zeros_like(theta)
        eps = 1e-5

        # Finite difference gradient
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_minus = theta.copy()

            theta_plus[i] += eps
            theta_minus[i] -= eps

            grad = (loss(x, theta_plus, y_true) - loss(x, theta_minus, y_true)) / (2 * eps)
            grads[i] = grad

        # Update parameters
        theta -= lr * grads

        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss(x, theta, y_true):.4f}")

    return theta


# ==============================
# 7. Train the Model
# ==============================

y_target = 1.0  # Desired output

theta_trained = train(x, y_target, theta.copy())

print("Trained theta:", theta_trained)
print("Final prediction:", predict(x, theta_trained))