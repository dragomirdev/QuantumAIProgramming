# ==========================================
# Quantum Classifier with SPSA Optimization
# ==========================================

import numpy as np
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

    # Trainable layer
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
    return (1 - exp_val) / 2


# ==========================================
# 3. Dataset
# ==========================================

X = np.array([
    [0.1, 0.2],
    [0.2, 0.3],
    [1.0, 1.2],
    [1.1, 1.3]
])

Y = np.array([0, 0, 1, 1])


# ==========================================
# 4. Loss Function
# ==========================================

def loss_fn(theta):
    preds = [predict(x, theta) for x in X]
    return np.mean((np.array(preds) - Y) ** 2)


# ==========================================
# 5. SPSA Step (Your Function)
# ==========================================

def spsa_step(theta, loss_fn, alpha=0.01, c=0.1):
    delta = np.random.choice([-1, 1], size=len(theta))

    loss_plus = loss_fn(theta + c * delta)
    loss_minus = loss_fn(theta - c * delta)

    grad = (loss_plus - loss_minus) / (2 * c * delta)

    return theta - alpha * grad


# ==========================================
# 6. Training Loop
# ==========================================

def train_spsa(theta, steps=100):
    for step in range(steps):
        theta = spsa_step(theta, loss_fn)

        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss_fn(theta):.6f}")

    return theta


# ==========================================
# 7. Initialize Parameters
# ==========================================

theta = np.random.randn(2)

print("Initial Loss:", loss_fn(theta))


# ==========================================
# 8. Train Model
# ==========================================

theta_trained = train_spsa(theta.copy())

print("\nTrained Parameters:", theta_trained)


# ==========================================
# 9. Evaluation
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