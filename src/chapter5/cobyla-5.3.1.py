# ==========================================
# Quantum Classifier with COBYLA Optimization
# ==========================================

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# ==========================================
# 1. Quantum Circuit Definition
# ==========================================

def circuit(x, theta):
    qc = QuantumCircuit(2)

    # Feature encoding
    qc.ry(x[0], 0)
    qc.ry(x[1], 1)

    # Entanglement
    qc.cx(0, 1)

    # Trainable parameters
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)

    # Simulate state
    state = Statevector.from_instruction(qc)

    # Z ⊗ I observable
    Z = np.kron([[1, 0], [0, -1]], np.eye(2))

    return np.real(state.expectation_value(Z))


# ==========================================
# 2. Prediction Function
# ==========================================

def predict(x, theta):
    exp_val = circuit(x, theta)
    return (1 - exp_val) / 2  # Convert to probability


# ==========================================
# 3. Loss Function
# ==========================================

def loss_fn(theta, X, Y):
    preds = [predict(x, theta) for x in X]
    return np.mean((np.array(preds) - Y) ** 2)


# ==========================================
# 4. COBYLA Training Function
# ==========================================

def train_cobyla(X, Y):
    init_theta = np.random.randn(2)

    result = minimize(
        lambda t: loss_fn(t, X, Y),
        x0=init_theta,
        method="COBYLA"
    )

    return result.x


# ==========================================
# 5. Dataset
# ==========================================

X = np.array([
    [0.1, 0.2],
    [0.2, 0.3],
    [1.0, 1.2],
    [1.1, 1.3]
])

Y = np.array([0, 0, 1, 1])


# ==========================================
# 6. Train Model
# ==========================================

theta_trained = train_cobyla(X, Y)

print("Trained Parameters:", theta_trained)


# ==========================================
# 7. Evaluation
# ==========================================

print("\nPredictions:")

correct = 0

for x, y in zip(X, Y):
    prob = predict(x, theta_trained)
    pred = int(prob > 0.5)

    print(f"Input: {x}, True: {y}, Prob: {prob:.4f}, Pred: {pred}")

    if pred == y:
        correct += 1

accuracy = correct / len(Y)

print("\nAccuracy:", accuracy)