"""
Test Project: Mini Function Approximation
==========================================
Tests your full environment: NumPy, Pandas, Matplotlib, scikit-learn, PyGAD, TensorFlow (with M2 GPU)

This is a tiny version of what your dissertation explores:
- Define a target function f(x) = sin(x) * cos(0.5x)
- Approach 1: Approximate it with a small neural network (TensorFlow on GPU)
- Approach 2: Use PyGAD to evolve coefficients for a simple basis function model
- Compare both approaches visually
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pygad
import tensorflow as tf
import time

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)
print(f"NumPy:        {np.__version__}")
print(f"Pandas:       {pd.__version__}")
print(f"TensorFlow:   {tf.__version__}")
print(f"PyGAD:        {pygad.__version__}")
print(f"GPU devices:  {tf.config.list_physical_devices('GPU')}")
print()

# --- Generate Data ---
np.random.seed(42)
x = np.linspace(-2 * np.pi, 2 * np.pi, 500).reshape(-1, 1)
y = np.sin(x) * np.cos(0.5 * x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Store results
results = {}

# --- Approach 1: Neural Network (TensorFlow on GPU) ---
print("Training Neural Network on M2 GPU...")
start = time.time()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
history = model.fit(x_train, y_train, epochs=200, batch_size=32, verbose=0, validation_split=0.2)

nn_time = time.time() - start
y_pred_nn = model.predict(x, verbose=0)
nn_mse = mean_squared_error(y_test, model.predict(x_test, verbose=0))
results['Neural Network'] = {'MSE': nn_mse, 'Time': nn_time}
print(f"  MSE: {nn_mse:.6f}  |  Time: {nn_time:.2f}s")

# --- Approach 2: PyGAD Basis Function Evolution ---
print("\nEvolving coefficients with PyGAD...")
start = time.time()

# Basis: f(x) ≈ a0 + a1*sin(x) + a2*cos(x) + a3*sin(2x) + a4*cos(2x) + a5*x
def build_basis(x_flat):
    return np.column_stack([
        np.ones_like(x_flat),
        np.sin(x_flat),
        np.cos(x_flat),
        np.sin(2 * x_flat),
        np.cos(2 * x_flat),
        x_flat
    ])

basis_train = build_basis(x_train.flatten())
basis_all = build_basis(x.flatten())

def fitness_func(ga_instance, solution, solution_idx):
    prediction = basis_train @ solution
    mse = np.mean((prediction - y_train.flatten()) ** 2)
    return 1.0 / (mse + 1e-8)

ga = pygad.GA(
    num_generations=200,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=50,
    num_genes=6,
    init_range_low=-2,
    init_range_high=2,
    mutation_percent_genes=30,
    suppress_warnings=True
)
ga.run()

best_solution = ga.best_solution()[0]
ga_time = time.time() - start
y_pred_ga = basis_all @ best_solution
basis_test = build_basis(x_test.flatten())
ga_mse = mean_squared_error(y_test.flatten(), basis_test @ best_solution)
results['PyGAD Evolution'] = {'MSE': ga_mse, 'Time': ga_time}
print(f"  MSE: {ga_mse:.6f}  |  Time: {ga_time:.2f}s")
print(f"  Evolved coefficients: {np.round(best_solution, 4)}")

# --- Results Table (Pandas) ---
print("\n" + "=" * 60)
print("RESULTS COMPARISON")
print("=" * 60)
df = pd.DataFrame(results).T
df.index.name = 'Method'
print(df.to_string())

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Both approximations vs truth
axes[0].plot(x, y, 'k-', linewidth=2, label='Target: sin(x)·cos(0.5x)')
axes[0].plot(x, y_pred_nn, 'r--', linewidth=1.5, label=f'Neural Net (MSE={nn_mse:.4f})')
axes[0].plot(x, y_pred_ga, 'b--', linewidth=1.5, label=f'PyGAD (MSE={ga_mse:.4f})')
axes[0].legend(fontsize=8)
axes[0].set_title('Function Approximation Comparison')
axes[0].grid(True, alpha=0.3)

# Plot 2: Training loss curve
axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Neural Network Training')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MSE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Error distribution
nn_error = y.flatten() - y_pred_nn.flatten()
ga_error = y.flatten() - y_pred_ga.flatten()
axes[2].hist(nn_error, bins=30, alpha=0.6, label='NN Error', color='red')
axes[2].hist(ga_error, bins=30, alpha=0.6, label='PyGAD Error', color='blue')
axes[2].set_title('Error Distribution')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results.png', dpi=150)
plt.show()
print("\nPlot saved as results.png")
print("✅ All systems working!")
