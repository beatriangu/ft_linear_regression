import pandas as pd
import matplotlib.pyplot as plt
from functions import prepare_data, get_weights


def main() -> None:
    """Runs the program, plots raw and normalized dataset. If the weights are not null, we draw a red line."""
    data = pd.read_csv("./data.csv")
    _, ax = plt.subplots(1, 2, figsize=(12, 6))  # Set figsize for a better view

    # Plot raw dataset
    ax[0].scatter(data=data, x="km", y="price")
    ax[0].set_title("Car price vs Kilometers Line Regression")
    ax[0].set_xlabel("km")  # Set x-axis label
    ax[0].set_ylabel("price")  # Set y-axis label

    # Get weights
    theta0, theta1 = get_weights()
    print(f"theta0: {theta0}, theta1: {theta1}")  # Debugging line

    if theta0 != 0 and theta1 != 0:
        # Calculate regression line
        x = data["km"]
        y_pred = theta0 + theta1 * x

        # Plot regression line
        ax[0].plot(x, y_pred, 'r', label=f'Regression Line: y = {theta0:.2f} {theta1:.2f}x')
        ax[0].legend()  # Show the legend

    # Prepare and plot normalized dataset
    prepare_data(data)
    ax[1].set_title("Normalized dataset")
    ax[1].scatter(data=data, x="km", y="price")
    ax[1].set_xlabel("km")  # Set x-axis label
    ax[1].set_ylabel("price")  # Set y-axis label
    
    plt.tight_layout()  # Adjust the layout
    plt.show()

if __name__ == "__main__":
    main()
