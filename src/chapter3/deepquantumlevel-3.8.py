# ==========================================
# Deep Quantum Model (3-layer circuit)
# ==========================================

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ==========================================
# 1. Deep Quantum Circuit
# ==========================================

def deep_model(x, theta):
    qc = QuantumCircuit(2)

    for i in range(3):  # 3 layers

        # Feature encoding
        qc.ry(x[0], 0)
        qc.ry(x[1], 1)

        # Entanglement
        qc.cx(0, 1)

        # Trainable layer
        qc.ry(theta[2 * i], 0)
        qc.ry(theta[2 * i + 1], 1)

    return qc


# ==========================================
# 2. Expectation Value Function
# ==========================================

def expectation_z0(qc):
    state = Statevector.from_instruction(qc)

    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    Z0 = np.kron(Z, I)

    return np.real(state.expectation_value(Z0))


# ==========================================
# 3. Prediction Function
# ==========================================

def predict(x, theta):
    qc = deep_model(x, theta)
    exp_val = expectation_z0(qc)

    # Convert to probability
    prob = (1 - exp_val) / 2
    return prob


# ==========================================
# 4. Loss Function (MSE)
# ==========================================

def loss(x, theta, y_true):
    y_pred = predict(x, theta)
    return (y_pred - y_true) ** 2


# ==========================================
# 5. Training Loop (Finite Difference)
# ==========================================

def train(x, y_true, theta, lr=0.1, steps=100):
    for step in range(steps):
        grads = np.zeros_like(theta)
        eps = 1e-5

        # Compute gradients
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
            print(f"Step {step}: Loss = {loss(x, theta, y_true):.6f}")

    return theta


# ==========================================
# 6. Example Input
# ==========================================

x = np.array([0.3, 0.8])  # Input features

# 3 layers × 2 parameters per layer = 6 parameters
theta = np.random.randn(6)

y_target = 1.0  # desired output

# ==========================================
# 7. Run Training
# ==========================================

print("Initial prediction:", predict(x, theta))

theta_trained = train(x, y_target, theta.copy())

print("\nTrained theta:", theta_trained)
print("Final prediction:", predict(x, theta_trained))