import numpy as np
import torch
from sklearn.linear_model import LinearRegression

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype = 'float32')
# Target (apples)
targets = np.array([[56], [81], [119], [22], [103]], dtype = 'float32')

# Convert inputs and targets to tensors
input_tensor = torch.tensor(inputs, requires_grad = True)
print(input_tensor)

target_tensor = torch.tensor(targets, requires_grad = True)
print(target_tensor)

from sklearn import metrics
import matplotlib.pyplot as plt

# Weights and biases
w = np.array([0.2, 0.5, 0.3])
b = 0.1
y = np.matmul(w, input_tensor.detach().numpy().T) + b

# Define the model
lr = LinearRegression()
lr.fit(input_tensor.detach().numpy(), y)

# Generate predictions
y_pred = lr.predict(input_tensor.detach().numpy())
print(y_pred)

# Compare with targets
print('Mean Absolute Error:', metrics.mean_absolute_error(target_tensor.detach().numpy(), y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(target_tensor.detach().numpy(), y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(target_tensor.detach().numpy(), y_pred)))

def MeanSquareError(t1, t2):
    diff = t1 - t2
    return np.sum(diff * diff) / diff.size

# Compute gradients
def model(x, w):
    return x @ w.T

def gradient_descent(X, y, w, learning_rate, n_iters):
    J_history = np.zeros((n_iters, 1))
    for i in range(n_iters):
        h = model(X, w)
        diff = h - y
        delta = (learning_rate / y.size) * (X.T @ diff)
        new_w = w - delta.T
        w = new_w
        J_history[i] = MeanSquareError(h, y)
    
    return (J_history, w)

# Compute loss
preds = model(input_tensor.detach().numpy(), w)
cost_initial = MeanSquareError(preds, target_tensor.detach().numpy())
print("Cost before regression: ", cost_initial)

# Train for 100 epochs
n_iters = 500
learning_rate = 0.01

initial_cost = MeanSquareError(model(input_tensor.detach().numpy(), w), target_tensor.detach().numpy())
print("Initial cost is: ", initial_cost)

(J_history, optimal_params) = gradient_descent(input_tensor.detach().numpy(), target_tensor.detach().numpy(), w, learning_rate, n_iters)
print("Optimal parameters are: ", optimal_params)
print("Final cost is: ", J_history[-1])

plt.plot(range(len(J_history)), J_history, 'r')
plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()