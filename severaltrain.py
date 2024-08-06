
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import prepare_data, predict, save_weights
from typing import Tuple

def error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Returns the error between our prediction and the actual price."""
    return prediction - target

def train(features: np.ndarray, targets: np.ndarray, theta0: float, theta1: float, learning_rate: float) -> Tuple[float, float]:
    """Train the weights with given features and targets, returns updated weights."""
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

def main() -> None:
    """Run the training and save the weights."""
    data = pd.read_csv("./data.csv")
    max_km, max_price = prepare_data(data)
    data = data.values
    features = data[:, 0]
    targets = data[:, 1]
    learning_rates = [0.01, 0.1, 0.5]
    epochs = 1000
    batch_size = data.shape[0]
    
    plt.figure(figsize=(10, 6))
    
    for learning_rate in learning_rates:
        errors = []
        theta0, theta1 = 0.0, 0.0
        
        for epoch in range(1, epochs + 1):
            for b in range(0, data.shape[0], batch_size):
                theta0, theta1 = train(features[b:b + batch_size], targets[b:b + batch_size], theta0, theta1, learning_rate)
            
            avg_error = cost(features, targets, theta0, theta1)
            errors.append(avg_error)
            print(f"Epoch {epoch:4}/{epochs:4}, L.Rate = {learning_rate}, average error: {avg_error:.6f}")
        
        # Plot errors for current learning rate
        plt.plot(np.array(errors), label=f"L.Rate = {learning_rate}")
    
    plt.title("Cost = f(epochs)")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
    