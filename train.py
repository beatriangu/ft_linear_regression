"""
This program finds the best theta0 and theta1 to best fit the dataset.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # Cambia el backend a Qt5Agg
import matplotlib.pyplot as plt

from functions import prepare_data, predict, save_weights
from typing import Tuple


def error(prediction: np.ndarray, target: np.ndarray) -> float:
    """Returns the error between our prediction and the actual price."""
    return prediction - target


def train(features: np.ndarray, targets: np.ndarray, theta0: float, theta1: float, learning_rate: float) -> Tuple[float, float]:
    """Train the weights with original data and returns them updated."""
    predictions = predict(theta0, theta1, features)
    errors = error(predictions, targets)
    delta0 = learning_rate * (1 / errors.shape[0]) * np.sum(errors)
    delta1 = learning_rate * (1 / errors.shape[0]) * np.sum(errors * features)
    return theta0 - delta0, theta1 - delta1


def cost(features: np.ndarray, targets: np.ndarray, theta0: float, theta1: float) -> float:
    """Return the average error for given weights."""
    predictions = predict(theta0, theta1, features)
    errors = np.abs(error(predictions, targets))
    return (1 / errors.shape[0]) * np.sum(errors)


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate the R^2 score.It gives you a measure of the percentage of accuracy or fit of the model to the data."""
    mean_actual = np.mean(actual)
    ss_total = np.sum((actual - mean_actual) ** 2)
    ss_residual = np.sum((actual - predicted) ** 2)
    
    r2 = 1 - (ss_residual / ss_total)
    return r2


def main() -> None:
    """Run the training and save the weights."""
    data = pd.read_csv("./data.csv")
    max_km, max_price = prepare_data(data)
    data = data.values
    features = data[:, 0]
    targets = data[:, 1]

    learning_rate = 0.1
    epochs = 1000
    batch_size = 32
    errors = []
    theta0, theta1 = 0.0, 0.

    for epoch in range(1, epochs + 1):
        for b in range(0, data.shape[0], batch_size):
            end = b + batch_size if b + batch_size <= data.shape[0] else data.shape[0]
            theta0, theta1 = train(features[b:end], targets[b:end], theta0, theta1, learning_rate)
       
        avg_error = cost(features, targets, theta0, theta1)
        errors.append(avg_error)
        print(f"Epoch {epoch:4}/{epochs:4}, average error: {avg_error:.6f}")

    # for epoch in range(1, epochs + 1):
    #     for b in range(0, data.shape[0], batch_size):
    #         end = b + batch_size if b + batch_size <= data.shape[0] else data.shape[0]
    #         theta0_temp, theta1_temp = train(features[b:end], targets[b:end], theta0, theta1, learning_rate)
            
    #         # Print temporary values
    #         print(f"Temporary values - Theta0: {theta0_temp:.6f}, Theta1: {theta1_temp:.6f}")
            
    #     # Assign temporary values to theta0 and theta1
    #     theta0, theta1 = theta0_temp, theta1_temp

    #     # Calculate and print average error for the epoch
    #     avg_error = cost(features, targets, theta0, theta1)
    #     errors.append(avg_error)
    #     print(f"Epoch {epoch:4}/{epochs:4}, average error: {avg_error:.6f}")

    # Plot errors
    plt.plot(np.array(errors))
    plt.title(f"Cost = f(epochs) | L.Rate = {learning_rate}")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.legend("")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot predictions vs actual data
    predictions = predict(theta0, theta1, features)
    r2 = r_squared(targets, predictions)
    print(f"R^2 Score: {r2:.4f} (Percentage of Precision: {r2 * 100:.2f}%)")
    plt.figure(figsize=(10, 6))
    
    plt.scatter(features, targets, label='Actual Data')
    plt.plot(features, predictions, color='red', label='Predictions')
    plt.title("Predictions vs Actual Data ")
    plt.xlabel("Kilometers")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Convert theta values to original scale and save weights
    theta0 *= max_price
    theta1 *= (max_price / max_km)
    print(f"Theta0: {theta0:.4f}")
    print(f"Theta1: {theta1:.4f}")
    save_weights(theta0, theta1)

if __name__ == "__main__":
    main()
