# ==========================================
# Quantum Mini-Batch Training + Shot Scheduling
# ==========================================

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

sim = AerSimulator()

# ==========================================
# 1. Quantum Circuit
# ==========================================

def circuit(x, theta):
    qc = QuantumCircuit(1, 1)

    qc.ry(x[0], 0)        # Feature encoding
    qc.ry(theta[0], 0)    # Trainable parameter

    qc.measure(0, 0)

    return qc


# ==========================================
# 2. Shot-Based Evaluation
# ==========================================

def shot_evaluation(qc, shots):
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    p1 = counts.get('1', 0) / shots
    return 1 - 2 * p1


# ==========================================
# 3. Prediction
# ==========================================

def predict(x, theta, shots):
    qc = circuit(x, theta)
    exp_val = shot_evaluation(qc, shots)
    return (1 - exp_val) / 2


# ==========================================
# 4. Dataset
# ==========================================

X = np.array([[0.1], [0.2], [1.0], [1.2], [0.15], [1.05]])
Y = np.array([0, 0, 1, 1, 0, 1])


# ==========================================
# 5. Adaptive Shot Scheduling
# ==========================================

def adaptive_shots(step, max_steps, min_shots=100, max_shots=2000):
    return int(min_shots + (max_shots - min_shots) * (step / max_steps))


# ==========================================
# 6. Mini-Batch Loss
# ==========================================

def batch_loss(theta, X, Y, batch_size, shots):
    idx = np.random.choice(len(X), batch_size)

    preds = [predict(X[i], theta, shots) for i in idx]
    targets = [Y[i] for i in idx]

    return np.mean((np.array(preds) - np.array(targets))**2)


# ==========================================
# 7. Gradient Estimation
# ==========================================

def compute_gradients(theta, X, Y, batch_size, shots):
    grads = np.zeros_like(theta)
    eps = 0.05  # larger epsilon helps with shot noise

    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[i] += eps
        theta_minus[i] -= eps

        loss_plus = batch_loss(theta_plus, X, Y, batch_size, shots)
        loss_minus = batch_loss(theta_minus, X, Y, batch_size, shots)

        grads[i] = (loss_plus - loss_minus) / (2 * eps)

    return grads


# ==========================================
# 8. Training Loop
# ==========================================

def train(theta, X, Y, steps=100, batch_size=3, lr=0.1):

    loss_history = []

    for step in range(steps):

        shots = adaptive_shots(step, steps)

        grads = compute_gradients(theta, X, Y, batch_size, shots)

        theta -= lr * grads

        loss_val = batch_loss(theta, X, Y, batch_size, shots)
        loss_history.append(loss_val)

        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss_val:.6f}, Shots = {shots}")

    return theta, loss_history


# ==========================================
# 9. Initialize Parameters
# ==========================================

theta = np.random.randn(1)

print("Initial Loss:", batch_loss(theta, X, Y, 3, 200))


# ==========================================
# 10. Train Model
# ==========================================

theta_trained, loss_history = train(theta.copy(), X, Y)

print("\nTrained Parameters:", theta_trained)


# ==========================================
# 11. Evaluation
# ==========================================

print("\nPredictions:")

correct = 0

for x, y in zip(X, Y):
    prob = predict(x, theta_trained, shots=2000)
    pred = int(prob > 0.5)

    print(f"Input: {x}, True: {y}, Prob: {prob:.4f}, Pred: {pred}")

    if pred == y:
        correct += 1

accuracy = correct / len(Y)

print("\nAccuracy:", accuracy)