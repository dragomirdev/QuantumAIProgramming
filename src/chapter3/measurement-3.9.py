# ==========================================
# Full Quantum AI Program (2-Qubit Classifier)
# ==========================================

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ==========================================
# 1. Quantum Model Definition
# ==========================================

def two_qubit_model(x, theta):
    qc = QuantumCircuit(2)

    # Feature encoding
    qc.ry(x[0], 0)
    qc.ry(x[1], 1)

    # Entanglement
    qc.cx(0, 1)

    # Trainable layer
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)

    return qc


# ==========================================
# 2. Prediction Function (YOUR FUNCTION)
# ==========================================

def predict(x, theta):
    qc = two_qubit_model(x, theta)
    state = Statevector.from_instruction(qc)

    Z0 = np.kron([[1, 0], [0, -1]], np.eye(2))

    return np.real(state.expectation_value(Z0))


# ==========================================
# 3. Convert Expectation → Probability
# ==========================================

def predict_proba(x, theta):
    exp_val = predict(x, theta)
    return (1 - exp_val) / 2


# ==========================================
# 4. Dataset (Simple Toy Data)
# ==========================================

# Binary classification: 0 or 1
X = np.array([
    [0.1, 0.2],
    [0.2, 0.3],
    [1.0, 1.2],
    [1.1, 1.3]
])

Y = np.array([0, 0, 1, 1])


# ==========================================
# 5. Loss Function (Mean Squared Error)
# ==========================================

def loss(theta):
    preds = []

    for x in X:
        p = predict_proba(x, theta)
        preds.append(p)

    preds = np.array(preds)
    return np.mean((preds - Y) ** 2)


# ==========================================
# 6. Training Loop (Finite Differences)
# ==========================================

def train(theta, lr=0.1, steps=100):
    for step in range(steps):
        grads = np.zeros_like(theta)
        eps = 1e-5

        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_minus = theta.copy()

            theta_plus[i] += eps
            theta_minus[i] -= eps

            grad = (loss(theta_plus) - loss(theta_minus)) / (2 * eps)
            grads[i] = grad

        # Update parameters
        theta -= lr * grads

        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss(theta):.6f}")

    return theta


# ==========================================
# 7. Initialize Parameters
# ==========================================

theta = np.random.randn(2)

print("Initial loss:", loss(theta))

# ==========================================
# 8. Train Model
# ==========================================

theta_trained = train(theta.copy())

print("\nTrained theta:", theta_trained)

# ==========================================
# 9. Evaluate Model
# ==========================================

print("\nPredictions:")
for x in X:
    prob = predict_proba(x, theta_trained)
    pred_class = int(prob > 0.5)
    print(f"Input: {x}, Prob: {prob:.4f}, Class: {pred_class}")