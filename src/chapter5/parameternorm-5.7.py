# ==========================================
# Quantum Training Monitoring (Section 5.7)
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# ==========================================
# 1. Quantum Circuit
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

    # Simulate
    state = Statevector.from_instruction(qc)

    # Observable Z ⊗ I
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    Z0 = np.kron(Z, I)

    return np.real(state.expectation_value(Z0))


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
# 5. Gradient Computation (Finite Difference)
# ==========================================

def compute_gradients(theta):
    grads = np.zeros_like(theta)
    eps = 1e-5

    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[i] += eps
        theta_minus[i] -= eps

        loss_plus = loss_fn(theta_plus)
        loss_minus = loss_fn(theta_minus)

        grads[i] = (loss_plus - loss_minus) / (2 * eps)

    return grads


# ==========================================
# 6. Training with Monitoring
# ==========================================

def train(theta, steps=100, lr=0.1):

    loss_history = []
    grad_norms = []
    param_norms = []

    for step in range(steps):

        grads = compute_gradients(theta)

        # Track metrics
        loss_val = loss_fn(theta)
        loss_history.append(loss_val)

        grad_norms.append(np.linalg.norm(grads))
        param_norms.append(np.linalg.norm(theta))

        # Update parameters
        theta -= lr * grads

        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss_val:.6f}, GradNorm = {grad_norms[-1]:.6f}")

    return theta, loss_history, grad_norms, param_norms


# ==========================================
# 7. Initialize Parameters
# ==========================================

theta = np.random.randn(2)

print("Initial Loss:", loss_fn(theta))


# ==========================================
# 8. Train Model
# ==========================================

theta_trained, loss_history, grad_norms, param_norms = train(theta.copy())

print("\nTrained Parameters:", theta_trained)


# ==========================================
# 9. Visualization of All Metrics
# ==========================================

plt.figure(figsize=(15, 4))

# Loss
plt.subplot(1, 3, 1)
plt.plot(loss_history)
plt.title("Loss Curve")
plt.xlabel("Step")
plt.ylabel("Loss")

# Gradient Norm
plt.subplot(1, 3, 2)
plt.plot(grad_norms)
plt.title("Gradient Norm")
plt.xlabel("Step")
plt.ylabel("||∇L||")

# Parameter Norm
plt.subplot(1, 3, 3)
plt.plot(param_norms)
plt.title("Parameter Norm")
plt.xlabel("Step")
plt.ylabel("||θ||")

plt.tight_layout()
plt.show()


# ==========================================
# 10. Evaluation
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